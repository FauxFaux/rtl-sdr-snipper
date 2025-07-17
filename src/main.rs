mod fft;

use crate::fft::SimpleFft;
use log::{LevelFilter, info};
use rtlsdr_rs::{DEFAULT_BUF_LENGTH, RtlSdr, error::Result};
use std::collections::VecDeque;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::{fs, io, process, thread};

const FREQUENCY: u32 = 434_200_000;
const SAMPLE_RATE: u32 = 2_880_000;

const DEBUG: bool = false;

// RTL Device Index
const RTL_INDEX: usize = 0;

fn main() {
    pretty_env_logger::formatted_builder()
        .filter_level(LevelFilter::Info)
        .init();

    // Shutdown flag that is set true when ctrl-c signal caught
    static SHUTDOWN: AtomicBool = AtomicBool::new(false);
    ctrlc::set_handler(|| {
        if SHUTDOWN.load(Ordering::Relaxed) {
            info!("Shutdown already requested, exiting immediately.");
            process::exit(1);
        }
        SHUTDOWN.swap(true, Ordering::Relaxed);
    })
    .unwrap();

    // Get radio and demodulation settings for given frequency and sample rate
    let radio_config = optimal_settings(FREQUENCY, SAMPLE_RATE);

    // Channel to pass receive data from receiver thread to processor thread
    let (tx, rx) = mpsc::channel();

    // Spawn thread to receive data from Radio
    let receive_thread = thread::spawn(|| receive(&SHUTDOWN, radio_config, tx));
    // Spawn thread to process data and output to stdout
    let process_thread = thread::spawn(|| process(&SHUTDOWN, rx));

    // Wait for threads to finish
    process_thread.join().unwrap();
    receive_thread.join().unwrap();
}

/// Thread to open SDR device and send received data to the demod thread until
/// SHUTDOWN flag is set to true.
fn receive(
    shutdown: &AtomicBool,
    radio_config: RadioConfig,
    tx: Sender<Box<[u8; DEFAULT_BUF_LENGTH]>>,
) {
    // Open device
    let mut sdr = RtlSdr::open(RTL_INDEX).expect("Failed to open device");
    // Config receiver
    config_sdr(
        &mut sdr,
        radio_config.capture_freq,
        radio_config.capture_rate,
    )
    .unwrap();

    info!("Tuned to {} Hz.\n", sdr.get_center_freq());
    info!(
        "Buffer size: {}ms",
        1000.0 * 0.5 * DEFAULT_BUF_LENGTH as f32 / radio_config.capture_rate as f32
    );
    info!("Sampling at {} S/s", sdr.get_sample_rate());

    info!("Reading samples in sync mode...");
    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        let mut buf: Box<[u8; DEFAULT_BUF_LENGTH]> = Box::new([0; DEFAULT_BUF_LENGTH]);
        let n = sdr.read_sync(&mut *buf);
        if n.is_err() {
            info!("Read error: {n:#?}");
            break;
        }
        let len = n.unwrap();
        if len < DEFAULT_BUF_LENGTH {
            info!("Short read ({len:#?}), samples lost, exiting!");
            break;
        }
        // Send received data through the channel to the processor thread
        tx.send(buf).expect("failed to send");
    }
    // Shut down the device and exit
    info!("Close");
    sdr.close().unwrap();
}

fn process(shutdown: &AtomicBool, rx: Receiver<Box<[u8; DEFAULT_BUF_LENGTH]>>) {
    let mut fft = SimpleFft::new(128);

    let mut buffer = VecDeque::with_capacity(64);

    while !shutdown.load(Ordering::Relaxed) {
        let buf = rx.recv().unwrap();
        let mut interesting_in_this_buf = 0;
        for chunk in buf.chunks_exact(2 * fft.len) {
            let interestingness = estimate_interestingness(&mut fft, chunk);
            let interesting = interestingness > 3.;
            if interesting {
                interesting_in_this_buf += 1;
                continue;
            }
        }

        let gap = 15;
        let currently_uninteresting = buffer.len() > gap
            && buffer
                .iter()
                .rev()
                .take(gap)
                .all(|(interestingness, _)| *interestingness == 0);

        if currently_uninteresting {
            buffer.pop_front();
        }
        buffer.push_back((interesting_in_this_buf, buf));

        let interesting_events = buffer
            .iter()
            .filter(|(interestingness, _)| *interestingness > 1)
            .count();

        if currently_uninteresting && interesting_events > 1 {
            write_out(buffer.iter().map(|(_, buf)| buf.as_slice()))
                .expect("writing buffer to file");
            info!(
                "Wrote {interesting_events}/{} interesting chunks to file",
                buffer.len()
            );
            buffer.truncate(0);
        }
    }
}

fn write_out<'v>(buffer: impl Iterator<Item = &'v [u8]>) -> io::Result<()> {
    let now = time::UtcDateTime::now()
        .format(&time::format_description::well_known::Rfc3339)
        .expect("well-known format")
        .replace(':', "_");

    let name = format!("snipper_{now}_{FREQUENCY}_{SAMPLE_RATE}.cu8");
    info!("Writing output to {name}");
    let mut file = fs::File::create(name)?;
    for buf in buffer {
        file.write_all(buf.as_ref())?;
    }

    file.flush()
}

fn estimate_interestingness(fft: &mut SimpleFft, chunk: &[u8]) -> f32 {
    let chunk = fft.process(chunk);
    let mut sorted = chunk.clone();
    sorted.sort_unstable_by(f32::total_cmp);
    assert_eq!(sorted.len(), fft.len);
    let low_estimate = sorted[sorted.len() * 75 / 100];
    let high_estimate = sorted[sorted.len() * 95 / 100];
    if DEBUG {
        debug_print(&chunk, &sorted);
    }

    high_estimate / low_estimate
}

fn debug_print(chunk: &[f32], sorted: &[f32]) {
    let low_estimate = sorted[sorted.len() * 75 / 100];
    let high_estimate = sorted[sorted.len() * 95 / 100];
    let ratio = high_estimate / low_estimate;
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let spark_chars = " ▁▂▃▄▅▆▇";
    let histo = sorted
        .iter()
        .step_by(sorted.len() / 10)
        .map(|v| {
            let pos = ((v - min) / (max - min) * (spark_chars.len() - 1) as f32).floor() as usize;
            spark_chars.chars().nth(pos).unwrap_or('X')
        })
        .collect::<String>();
    println!(
        "median: {:.2} 90%: {:.2}, ratio: {:.2}, {} {}",
        low_estimate,
        high_estimate,
        ratio,
        histo,
        chunk
            .iter()
            .map(|v| {
                let pos =
                    ((v - min) / (max - min) * (spark_chars.len() - 1) as f32).floor() as usize;
                spark_chars.chars().nth(pos).unwrap_or('X')
            })
            .collect::<String>()
    );
}

/// Radio configuration produced by `optimal_settings`
struct RadioConfig {
    capture_freq: u32,
    capture_rate: u32,
}

/// Determine the optimal radio and demodulation configurations for given
/// frequency and sample rate.
fn optimal_settings(freq: u32, rate: u32) -> RadioConfig {
    let downsample = (1_000_000 / rate) + 1;
    info!("downsample: {downsample}");
    let capture_rate = downsample * rate;
    info!("rate_in: {rate} capture_rate: {capture_rate}");
    // Use offset-tuning
    let capture_freq = freq + capture_rate / 4;
    info!("capture_freq: {capture_freq}");

    RadioConfig {
        capture_freq,
        capture_rate,
    }
}

/// Configure the SDR device for a given receive frequency and sample rate.
fn config_sdr(sdr: &mut RtlSdr, freq: u32, rate: u32) -> Result<()> {
    // Use auto-gain
    sdr.set_tuner_gain(rtlsdr_rs::TunerGain::Auto)?;
    // Disable bias-tee
    sdr.set_bias_tee(false)?;
    // Reset the endpoint before we try to read from it (mandatory)
    sdr.reset_buffer()?;
    // Set the frequency
    sdr.set_center_freq(freq)?;
    // Set sample rate
    sdr.set_sample_rate(rate)?;
    Ok(())
}
