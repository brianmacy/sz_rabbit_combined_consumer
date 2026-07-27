//! Shared binary runtime helpers used by every backend driver (RabbitMQ, SQS).
//!
//! Each backend's `main()` parses its own CLI args and dispatches; the AMQP /
//! SQS load loops live in their respective bins. But everything here — logging
//! init, environment bring-up, bounded native teardown, and the two
//! backend-agnostic run modes (pure redoer, file loader) — is identical across
//! backends, so it lives here to avoid duplication.

use std::io::Write;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::Duration;

use sz_rust_sdk::prelude::*;
use tracing_subscriber::{EnvFilter, fmt};

use crate::config::Config;
use crate::stats::{self, RUNNING};
use crate::{file_loader, pure_redoer};

/// Upper bound on the native environment teardown at shutdown.
///
/// `SzEnvironmentCore::destroy()` calls `Sz_destroy()`, an
/// uninterruptible native FFI call with no timeout that can BLOCK indefinitely on
/// the SIGTERM shutdown path once the worker threads that made engine calls have
/// exited. We run it on a dedicated thread and wait only up to this bound; past
/// it the process is exiting anyway, so we hard-exit and let the OS reclaim
/// native resources (no use-after-free — the whole process is gone). Kept well
/// under the sibling drivers' 10s worker-join grace and the 30s e2e test grace.
const TEARDOWN_GRACE: Duration = Duration::from_secs(5);

/// Installs the tracing subscriber (SENZING_LOG_LEVEL default, RUST_LOG override)
/// and anchors the process start time for throughput lines. Call once at startup.
pub fn init_logging() {
    let log_level = std::env::var("SENZING_LOG_LEVEL").unwrap_or_else(|_| "info".to_string());
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(map_log_level(&log_level)));
    fmt().with_env_filter(env_filter).with_target(false).init();
    let _ = stats::start_time();
}

/// Maps the Python-style SENZING_LOG_LEVEL names onto tracing levels.
fn map_log_level(level: &str) -> &'static str {
    match level.to_lowercase().as_str() {
        "notset" | "debug" => "debug",
        "warning" | "warn" => "warn",
        "error" => "error",
        "fatal" | "critical" => "error",
        _ => "info",
    }
}

/// Initializes the Senzing environment singleton (exactly one `Sz_init` per
/// process; every thread derives its own engine handle). On failure prints a loud
/// message and returns the process exit code the caller should return.
pub fn init_environment(
    instance_name: &str,
    config: &Config,
) -> Result<Arc<SzEnvironmentCore>, ExitCode> {
    match SzEnvironmentCore::get_instance(instance_name, &config.engine_config, config.debug_trace)
    {
        Ok(e) => Ok(e),
        Err(e) => {
            eprintln!("Failed to initialize Senzing environment: {e}");
            Err(ExitCode::from(255))
        }
    }
}

/// Installs the graceful-shutdown signal handler (SIGINT/SIGTERM/SIGHUP via the
/// ctrlc `termination` feature) that flips `RUNNING` to false.
pub fn install_shutdown_handler() {
    if let Err(e) = ctrlc::set_handler(|| {
        tracing::warn!("Graceful shutdown requested");
        RUNNING.store(false, Ordering::Relaxed);
    }) {
        tracing::warn!("Could not install signal handler: {e}");
    }
}

/// Tears the native Senzing environment down (best effort, time-bounded) and then
/// terminates the process with `code`, GUARANTEEING a prompt exit within the
/// SIGTERM grace regardless of whether `Sz_destroy()` / runtime drops would wedge.
///
/// Only ever called AFTER the workers/fetcher have joined (or been deliberately
/// detached) and stdout has the final totals, so exiting here loses nothing.
pub fn teardown_and_exit(code: u8) -> ! {
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let spawned = std::thread::Builder::new()
        .name("sz-teardown".to_string())
        .spawn(move || {
            // Ownership-based teardown (replaces the removed
            // `destroy_global_instance()`): reacquire the process singleton and
            // consume it. `destroy()` first removes the global reference, then
            // `Arc::try_unwrap`s — native `Sz_destroy()` only runs when this is
            // the SOLE remaining reference. We are only ever called AFTER all
            // worker/fetcher threads (and their `Arc<SzEnvironmentCore>` clones)
            // have joined, so this clone plus the singleton are the only two refs
            // and try_unwrap succeeds. If a clone somehow survived, destroy()
            // returns an error (logged) and safely skips the native call rather
            // than risking a use-after-free.
            match SzEnvironmentCore::get_existing_instance() {
                Ok(env) => {
                    if let Err(e) = env.destroy() {
                        tracing::warn!("error destroying Senzing environment: {e}");
                    }
                }
                Err(e) => tracing::warn!("no Senzing environment to destroy: {e}"),
            }
            let _ = done_tx.send(());
        });

    match spawned {
        // recv_timeout errors on both timeout AND a dropped sender (teardown
        // thread panicked); either way we have waited long enough.
        Ok(_) => match done_rx.recv_timeout(TEARDOWN_GRACE) {
            Ok(()) => {
                tracing::info!("Senzing environment destroyed; exiting cleanly");
                // Nothing is wedged by definition on this branch, so a normal exit
                // (atexit handlers, static destructors) is safe and preferred.
                flush_and_exit(code)
            }
            Err(_) => {
                tracing::warn!(
                    "native teardown did not complete within {TEARDOWN_GRACE:?}; forcing \
                     process exit (OS reclaims native resources)"
                );
                // The teardown thread is still inside Sz_destroy(): MUST NOT run atexit.
                flush_and_force_exit(code)
            }
        },
        Err(e) => {
            tracing::warn!("could not spawn teardown thread ({e}); forcing process exit");
            flush_and_force_exit(code)
        }
    }
}

/// Hard-exit WITHOUT attempting the native teardown (leak-on-exit): used when a
/// worker is still in an uninterruptible engine call, so `Sz_destroy()` would
/// risk a use-after-free / wedge. The OS reclaims everything.
pub fn leak_and_exit(code: u8) -> ! {
    flush_and_force_exit(code)
}

/// Normal exit: flush stdout, then `std::process::exit` (runs libc atexit handlers
/// and static destructors). Only for paths where no native thread can be stuck.
fn flush_and_exit(code: u8) -> ! {
    // Flush stdout so the final "Processed total ..." line the e2e tests scrape is
    // never lost to process::exit skipping Rust's buffered-writer drop.
    let _ = std::io::stdout().flush();
    std::process::exit(code as i32);
}

/// Exit that CANNOT be blocked by a wedged native thread.
///
/// ⚠ `std::process::exit` calls libc `exit(3)`, which runs atexit handlers and
/// static destructors. When a Senzing/ODBC thread is stuck mid-engine-call, one of
/// those handlers can block on a lock that thread holds — so `exit(3)` itself hangs,
/// in exactly the situation the caller invoked a "forced exit" to escape.
///
/// Observed in production 2026-07-27: 17 of 20 consumers on one host logged
/// "skipping Senzing environment destroy … forcing process exit" during a database
/// restart and then **never exited**. Because the process never terminated,
/// `RestartPolicy=on-failure` never fired: the containers stayed `Up`, RestartCount
/// stayed flat, their logs went silent, and the last-emitted stats blob kept
/// reporting a healthy `adds_rate`. The fleet ran at 57% capacity for 80 minutes and
/// every container-level health check reported it healthy — the only observable was
/// the host's AMQP consumer count (3 instead of 20).
///
/// `_exit(2)` terminates immediately without running atexit handlers or flushing
/// libc buffers, so it cannot be blocked. We flush our own stdout first, which is
/// the only buffered output that matters here.
fn flush_and_force_exit(code: u8) -> ! {
    let _ = std::io::stdout().flush();
    // SAFETY: `_exit` is async-signal-safe and never returns. It deliberately skips
    // atexit handlers / static destructors — that is the entire point of this path.
    unsafe { libc::_exit(code as i32) }
}

/// Applies the shared use-after-free exit discipline given `(workers_clean,
/// result)` from a run: destroy + exit on a clean join, else leak-on-exit.
fn finish(workers_clean: bool, result: anyhow::Result<()>) -> ! {
    let code = match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("{e:#}");
            255
        }
    };
    if workers_clean {
        teardown_and_exit(code);
    } else {
        tracing::warn!(
            "worker still in an uninterruptible engine call at shutdown; skipping \
             native teardown and forcing process exit (restart-on-failure restarts clean)"
        );
        leak_and_exit(code);
    }
}

/// redo% = 100: pure `std::thread` redoer (no AMQP/SQS, no tokio). Never returns.
pub fn run_pure_redoer(config: &Config, env: Arc<SzEnvironmentCore>) -> ! {
    install_shutdown_handler();
    let (workers_clean, result) = pure_redoer::run(config, env);
    finish(workers_clean, result)
}

/// File-input mode: pure `std::thread` loader reading JSONL from a single file
/// (no AMQP/SQS, no tokio). Never returns.
pub fn run_file_loader(config: &Config, env: Arc<SzEnvironmentCore>) -> ! {
    install_shutdown_handler();
    let (workers_clean, result) = file_loader::run(config, env);
    finish(workers_clean, result)
}
