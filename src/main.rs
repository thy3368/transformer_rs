fn main() {
    if let Err(error) = transformer_rs::adapter::inbound::cli::run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
