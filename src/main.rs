use std::io::Write;

use clap::Parser;
use minijinja::{context, Environment};
use ollama_rs::{generation::completion::request::GenerationRequest, IntoUrlSealed, Ollama};
use serde::Deserialize;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};
use tokio_stream::StreamExt;
use toml;

#[derive(Debug, Parser)]
struct Args {
    prompt: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Config {
    addr: String,
    model: String,
    color: String,
    prompt_template: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config_path = std::env::var("CONFIG").expect("Missing CONFIG");
    let config = parse_config(&config_path);

    let args = Args::parse();
    let question = args.prompt.join(" ");
    let prompt = parse_prompt(&question, &config.prompt_template);

    let url = config.addr.into_url().unwrap();
    let ollama = Ollama::from_url(url);
    let mut stream = ollama
        .generate_stream(GenerationRequest::new(config.model, prompt))
        .await
        .unwrap();

    let mut stdout = StandardStream::stdout(ColorChoice::Always);
    let color = parse_color(&config.color);
    stdout.set_color(ColorSpec::new().set_fg(Some(color)))?;

    while let Some(Ok(responses)) = stream.next().await {
        for resp in responses {
            stdout.write(resp.response.as_bytes())?;
            stdout.flush()?;
        }
    }

    stdout.write(b"\n")?;

    Ok(())
}

fn parse_prompt(question: &str, template: &str) -> String {
    let mut env = Environment::new();
    env.add_template("prompt_template", template).unwrap();
    let template = env.get_template("prompt_template").unwrap();
    let t = template.render(context!(question => question)).unwrap();
    t
}

fn parse_config(path: &str) -> Config {
    let file = std::fs::read_to_string(path).unwrap();
    let config: Config = toml::from_str(&file).unwrap();
    config
}

fn parse_color(color: &str) -> Color {
    let color = color.trim().to_lowercase();
    match color.as_str() {
        "purple" => Color::Rgb(202, 158, 230),
        "red" => Color::Rgb(231, 130, 132),
        "green" => Color::Rgb(166, 209, 137),
        _ => Color::Green,
    }
}
