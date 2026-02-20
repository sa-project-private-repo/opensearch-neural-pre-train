//! Fast IDF weight computation for SPLADE training.
//!
//! Replaces Python IDF computation (~47 min) with parallel Rust (~2-3 min).
//! Output is compatible with PyTorch: saves as raw f32 binary + JSON metadata.

use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::Deserialize;
use tokenizers::tokenizer::Tokenizer;

#[derive(Parser)]
#[command(name = "idf-compute", about = "Fast IDF computation for SPLADE")]
struct Args {
    /// Glob pattern for input JSONL files
    #[arg(short, long, default_value = "data/v29.0/train_shard_*.jsonl")]
    input: String,

    /// Output path (without extension)
    #[arg(short, long, default_value = "outputs/idf_weights/xlmr_v29_idf")]
    output: String,

    /// Path to tokenizer.json file
    #[arg(short, long)]
    tokenizer: String,

    /// IDF smoothing method: bm25 or standard
    #[arg(short, long, default_value = "bm25")]
    smoothing: String,

    /// Number of threads (0 = auto)
    #[arg(long, default_value = "0")]
    threads: usize,
}

#[derive(Deserialize)]
struct Record {
    query: Option<String>,
    positive: Option<String>,
    negative: Option<String>,
    text: Option<String>,
    document: Option<String>,
}

fn resolve_tokenizer_path(input: &str) -> PathBuf {
    let path = PathBuf::from(input);
    // Direct file path
    if path.exists() {
        return path;
    }
    // Directory containing tokenizer.json
    let dir_path = path.join("tokenizer.json");
    if dir_path.exists() {
        return dir_path;
    }
    // HuggingFace cache lookup
    let home = std::env::var("HOME").unwrap_or_default();
    let cache_base = format!("{}/.cache/huggingface/hub", home);
    let model_name = input.replace('/', "--");
    let models_dir = format!("{}/models--{}/snapshots", cache_base, model_name);
    if let Ok(entries) = fs::read_dir(&models_dir) {
        for entry in entries.flatten() {
            let candidate = entry.path().join("tokenizer.json");
            if candidate.exists() {
                return candidate;
            }
        }
    }
    eprintln!("Cannot find tokenizer.json for '{}'. Provide path to tokenizer.json file.", input);
    std::process::exit(1);
}

fn main() {
    let args = Args::parse();

    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .unwrap();
    }

    // Load tokenizer from file
    let tokenizer_path = resolve_tokenizer_path(&args.tokenizer);
    eprintln!("Loading tokenizer: {}...", tokenizer_path.display());
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .expect("Failed to load tokenizer");
    let vocab_size = tokenizer.get_vocab_size(true);
    eprintln!("Vocab size: {}", vocab_size);

    // Serialize tokenizer bytes for thread-local cloning
    let tokenizer_bytes = tokenizer.to_string(true).unwrap().into_bytes();

    // Find input files
    let mut files: Vec<PathBuf> = glob::glob(&args.input)
        .expect("Invalid glob pattern")
        .filter_map(|r| r.ok())
        .collect();
    files.sort();

    if files.is_empty() {
        eprintln!("No files found matching: {}", args.input);
        std::process::exit(1);
    }
    eprintln!("Found {} files", files.len());

    // Phase 1: Count document frequencies in parallel
    let total_docs = AtomicU64::new(0);
    let global_df: Mutex<Vec<u64>> = Mutex::new(vec![0u64; vocab_size]);

    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} files ({eta})")
            .unwrap(),
    );

    files.par_iter().for_each(|file_path| {
        // Each thread gets its own tokenizer instance
        let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes).unwrap();
        let mut local_df = vec![0u64; vocab_size];
        let mut local_docs = 0u64;

        let file = File::open(file_path).expect("Cannot open file");
        let reader = BufReader::with_capacity(1 << 20, file);

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            if line.is_empty() {
                continue;
            }

            let record: Record = match serde_json::from_str(&line) {
                Ok(r) => r,
                Err(_) => continue,
            };

            // Process each text field as a separate document
            let texts: Vec<&str> = [
                record.query.as_deref(),
                record.positive.as_deref(),
                record.negative.as_deref(),
                record.text.as_deref(),
                record.document.as_deref(),
            ]
            .into_iter()
            .flatten()
            .filter(|s| !s.is_empty())
            .collect();

            for text in texts {
                let encoding: tokenizers::Encoding = match tokenizer.encode(text, false) {
                    Ok(e) => e,
                    Err(_) => continue,
                };

                // Unique token IDs in this document
                let mut seen = vec![false; vocab_size];
                for &id in encoding.get_ids() {
                    let id = id as usize;
                    if id < vocab_size && !seen[id] {
                        seen[id] = true;
                        local_df[id] += 1;
                    }
                }
                local_docs += 1;
            }
        }

        // Merge local counts into global
        total_docs.fetch_add(local_docs, Ordering::Relaxed);
        let mut gdf = global_df.lock().unwrap();
        for i in 0..vocab_size {
            gdf[i] += local_df[i];
        }

        pb.inc(1);
    });

    pb.finish_with_message("Done");

    let n = total_docs.load(Ordering::Relaxed) as f64;
    let df = global_df.lock().unwrap();
    eprintln!("Total documents: {}", n as u64);

    // Phase 2: Compute IDF weights
    eprintln!("Computing IDF weights (smoothing={})...", args.smoothing);
    let mut idf_weights = vec![0.0f32; vocab_size];

    for i in 0..vocab_size {
        let doc_freq = df[i] as f64;
        idf_weights[i] = match args.smoothing.as_str() {
            "bm25" => ((1.0 + (n - doc_freq + 0.5) / (doc_freq + 0.5)).ln()) as f32,
            _ => ((n / (doc_freq + 1.0)).ln()) as f32,
        };
    }

    // Phase 3: Save output
    let output_path = PathBuf::from(&args.output);
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).unwrap();
    }

    // Save as raw f32 binary (PyTorch-compatible via numpy)
    let bin_path = output_path.with_extension("bin");
    let mut writer = BufWriter::new(File::create(&bin_path).unwrap());
    for &w in &idf_weights {
        writer.write_all(&w.to_le_bytes()).unwrap();
    }
    writer.flush().unwrap();
    eprintln!("Saved weights to {}", bin_path.display());

    // Save metadata
    let meta_path = output_path.with_extension("json");
    let metadata = serde_json::json!({
        "vocab_size": vocab_size,
        "num_docs": n as u64,
        "smoothing": args.smoothing,
        "tokenizer_name": args.tokenizer,
        "dtype": "float32",
        "byte_order": "little_endian",
    });
    let mut f = File::create(&meta_path).unwrap();
    f.write_all(serde_json::to_string_pretty(&metadata).unwrap().as_bytes())
        .unwrap();
    eprintln!("Saved metadata to {}", meta_path.display());

    // Stats
    let nonzero = idf_weights.iter().filter(|&&w| w > 0.0).count();
    let max_idf = idf_weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_idf = idf_weights.iter().cloned().fold(f32::INFINITY, f32::min);
    eprintln!(
        "Stats: nonzero={}/{} min={:.4} max={:.4}",
        nonzero, vocab_size, min_idf, max_idf
    );
}
