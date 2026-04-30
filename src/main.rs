// stoki-news-server
// VPS에서 실행. 15분마다 RSS 수집 + gemma4:e2b 요약 → HTTP API 제공.
//
// ┌─────────────────────────────────────────────────────────────────────┐
// │  ENV 변수                                                           │
// │  PORT           포트번호         (기본: 8765)                       │
// │  OLLAMA_HOST    Ollama 주소       (기본: http://localhost:11434)     │
// │  OLLAMA_MODEL   사용 모델         (기본: gemma4:e2b)                │
// │  FETCH_INTERVAL 갱신 주기(초)     (기본: 900 = 15분)                │
// │  SUMMARY_TIMEOUT 요약 제한(초)    (기본: 300)                      │
// └─────────────────────────────────────────────────────────────────────┘
//
// ┌─────────────────────────────────────────────────────────────────────┐
// │  API                                                                │
// │  GET /news     → 최신 뉴스 + 요약 JSON                             │
// │  GET /health   → "ok"                                               │
// └─────────────────────────────────────────────────────────────────────┘
//
// VPS 빌드: cargo build --release
// 실행:     PORT=8765 OLLAMA_MODEL=gemma4:e2b ./target/release/stoki-news-server

use axum::{
    extract::State,
    http::Method,
    response::Json,
    routing::get,
    Router,
};
use chrono::Utc;
use rss::Channel;
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, sync::Arc, time::{Duration, Instant}};
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};

// ---------------------------------------------------------------------------
// 공유 데이터 타입
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct NewsItem {
    pub title: String,
    pub source: String,
    pub pub_date: String,
    pub link: String,
    pub category: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct NewsResponse {
    pub date: String,
    pub fetched_at: String,
    pub summary: Option<String>,
    pub items: Vec<NewsItem>,
}

type SharedState = Arc<RwLock<NewsResponse>>;

// ---------------------------------------------------------------------------
// RSS 소스 목록
// ---------------------------------------------------------------------------

const STOCK_SOURCES: &[(&str, &str)] = &[
    (
        "Google Finance",
        "https://news.google.com/rss/search?q=stock+market+finance&hl=en-US&gl=US&ceid=US:en",
    ),
    ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
    (
        "CNBC",
        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    ),
    (
        "Nasdaq",
        "https://www.nasdaq.com/feed/rssoutbound?category=Market+Activity",
    ),
];

const CRYPTO_SOURCES: &[(&str, &str)] = &[
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("CoinTelegraph", "https://cointelegraph.com/rss"),
    (
        "Google Crypto",
        "https://news.google.com/rss/search?q=bitcoin+crypto+ethereum&hl=en-US&gl=US&ceid=US:en",
    ),
    ("Decrypt", "https://decrypt.co/feed"),
];

// ---------------------------------------------------------------------------
// RSS fetch
// ---------------------------------------------------------------------------

async fn fetch_all_rss(client: &reqwest::Client) -> Vec<NewsItem> {
    let mut all: Vec<NewsItem> = Vec::new();

    for &(src, url) in STOCK_SOURCES {
        match fetch_one(client, src, url, "stock").await {
            Ok(items) => {
                println!("[rss] {src} → {}개", items.len());
                all.extend(items);
            }
            Err(e) => eprintln!("[rss] {src} 실패: {e}"),
        }
    }
    for &(src, url) in CRYPTO_SOURCES {
        match fetch_one(client, src, url, "crypto").await {
            Ok(items) => {
                println!("[rss] {src} → {}개", items.len());
                all.extend(items);
            }
            Err(e) => eprintln!("[rss] {src} 실패: {e}"),
        }
    }

    all.sort_by(|a, b| b.pub_date.cmp(&a.pub_date));

    let mut seen_links = std::collections::HashSet::new();
    let mut seen_titles = std::collections::HashSet::new();
    all.retain(|item| {
        let lk = item.link.trim().to_lowercase();
        let tk: String = item
            .title
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric())
            .take(60)
            .collect();
        if !lk.is_empty() && !seen_links.insert(lk) {
            return false;
        }
        seen_titles.insert(tk)
    });

    all.truncate(100);
    all
}

async fn fetch_one(
    client: &reqwest::Client,
    source: &str,
    url: &str,
    category: &str,
) -> Result<Vec<NewsItem>, Box<dyn std::error::Error + Send + Sync>> {
    let text = client.get(url).send().await?.text().await?;
    let ch = text.parse::<Channel>()?;
    let items = ch
        .items()
        .iter()
        .map(|i| NewsItem {
            title: i.title().unwrap_or("(no title)").to_string(),
            source: source.to_string(),
            pub_date: i.pub_date().unwrap_or("").to_string(),
            link: i.link().unwrap_or("").to_string(),
            category: category.to_string(),
        })
        .collect();
    Ok(items)
}

// ---------------------------------------------------------------------------
// Ollama 요약
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct OllamaRequest<'a> {
    model: &'a str,
    prompt: String,
    system: &'a str,
    stream: bool,
    think: bool,
    keep_alive: &'a str,
    options: OllamaOptions,
}

#[derive(Serialize)]
struct OllamaOptions {
    temperature: f32,
    num_predict: u32,
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
    thinking: Option<String>,
    done: Option<bool>,
    total_duration: Option<u64>,
    load_duration: Option<u64>,
    prompt_eval_count: Option<u32>,
    eval_count: Option<u32>,
}

#[derive(Deserialize)]
struct OllamaError {
    error: String,
}

async fn summarize(client: &reqwest::Client, items: &[NewsItem]) -> Option<String> {
    if items.is_empty() {
        return None;
    }

    let ollama_host = std::env::var("OLLAMA_HOST")
        .unwrap_or_else(|_| "http://localhost:11434".to_string());
    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "gemma4:e2b".to_string());
    let timeout_secs = std::env::var("SUMMARY_TIMEOUT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let url = format!("{}/api/generate", ollama_host.trim_end_matches('/'));

    let mut list = String::new();
    for (i, it) in items.iter().take(5).enumerate() {
        list.push_str(&format!("{}. [{}] {}\n", i + 1, it.source, it.title));
    }

    let prompt = format!(
        "Below are the latest 5 financial news headlines.\n\
Summarize each in one short English sentence.\n\
Output must be exactly 5 bullet lines.\n\
Bullet format: \"- summary\"\n\
\nHeadlines:\n{list}"
    );

    let body = OllamaRequest {
        model: &model,
        prompt,
        system: "You summarize financial news headlines. Use only the headlines and do not speculate.",
        stream: false,
        think: false,
        keep_alive: "30m",
        options: OllamaOptions {
            temperature: 0.2,
            num_predict: 512,
        },
    };

    println!(
        "[summary] Ollama 요청 시작: model={model}, url={url}, timeout={timeout_secs}s"
    );

    let started = Instant::now();

    match client
        .post(&url)
        .json(&body)
        .timeout(Duration::from_secs(timeout_secs))
        .send()
        .await
    {
        Ok(resp) => {
            let status = resp.status();
            let text = match resp.text().await {
                Ok(t) => t,
                Err(e) => { eprintln!("[summary] 응답 읽기 실패: {e}"); return None; }
            };

            if !status.is_success() {
                match serde_json::from_str::<OllamaError>(&text) {
                    Ok(err) => eprintln!("[summary] Ollama 오류 HTTP {status}: {}", err.error),
                    Err(_) => eprintln!(
                        "[summary] Ollama 오류 HTTP {status}: {}",
                        &text[..text.len().min(500)]
                    ),
                }
                return None;
            }

            match serde_json::from_str::<OllamaResponse>(&text) {
                Ok(r) => {
                    let s = r.response.trim().to_string();
                    if s.is_empty() {
                        eprintln!(
                            "[summary] Ollama 빈 응답: elapsed={}s done={:?} total_ms={:?} load_ms={:?} prompt_tokens={:?} output_tokens={:?} thinking_chars={}",
                            started.elapsed().as_secs(),
                            r.done,
                            r.total_duration.map(|n| n / 1_000_000),
                            r.load_duration.map(|n| n / 1_000_000),
                            r.prompt_eval_count,
                            r.eval_count,
                            r.thinking.as_deref().unwrap_or("").chars().count(),
                        );
                        None
                    } else {
                        println!(
                            "[summary] Ollama 응답 완료: elapsed={}s chars={} output_tokens={:?}",
                            started.elapsed().as_secs(),
                            s.chars().count(),
                            r.eval_count,
                        );
                        Some(s)
                    }
                }
                Err(e) => {
                    eprintln!("[summary] 파싱 실패: {e}");
                    eprintln!("[summary] Ollama 응답: {}", text.chars().take(300).collect::<String>());
                    None
                }
            }
        }
        Err(e) => {
            eprintln!("[summary] Ollama 요청 실패 ({url}): {e}");
            None
        }
    }
}

// ---------------------------------------------------------------------------
// 갱신 루프
// ---------------------------------------------------------------------------

async fn refresh_loop(state: SharedState, interval_secs: u64) {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .user_agent("Mozilla/5.0 (compatible; StokiNewsServer/1.0)")
        .build()
        .expect("reqwest client");

    loop {
        println!("[server] RSS 수집 시작...");
        let items = fetch_all_rss(&client).await;
        println!("[server] 총 {}개 수집 완료. 요약 중...", items.len());

        let now = Utc::now();
        *state.write().await = NewsResponse {
            date: now.format("%Y-%m-%d").to_string(),
            fetched_at: now.to_rfc3339(),
            summary: None,
            items: items.clone(),
        };
        println!("[server] 뉴스 캐시 선갱신 완료. 요약 대기 중.");

        let summary = summarize(&client, &items).await;
        match &summary {
            Some(_) => println!("[server] 요약 완료."),
            None    => println!("[server] 요약 실패 — 뉴스만 제공."),
        }

        let now = Utc::now();
        let payload = NewsResponse {
            date: now.format("%Y-%m-%d").to_string(),
            fetched_at: now.to_rfc3339(),
            summary,
            items,
        };

        *state.write().await = payload;
        println!("[server] 캐시 갱신 완료. {interval_secs}초 후 재갱신.");

        tokio::time::sleep(Duration::from_secs(interval_secs)).await;
    }
}

// ---------------------------------------------------------------------------
// HTTP 핸들러
// ---------------------------------------------------------------------------

async fn handle_news(State(state): State<SharedState>) -> Json<NewsResponse> {
    let news = state.read().await.clone();
    println!("[server] /news 요청 → {}개 전송", news.items.len());
    Json(news)
}

async fn handle_health() -> &'static str {
    "ok"
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8765);

    let interval: u64 = std::env::var("FETCH_INTERVAL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(900);

    let summary_timeout: u64 = std::env::var("SUMMARY_TIMEOUT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);

    println!("[server] stoki-news-server 시작");
    println!("[server] 포트: {port}  갱신주기: {interval}초  요약제한: {summary_timeout}초");
    println!(
        "[server] Ollama: {}  모델: {}",
        std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string()),
        std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "gemma4:e2b".to_string()),
    );

    let state: SharedState = Arc::new(RwLock::new(NewsResponse::default()));

    {
        let s = state.clone();
        tokio::spawn(async move {
            refresh_loop(s, interval).await;
        });
    }

    let cors = CorsLayer::new()
        .allow_methods([Method::GET])
        .allow_origin(Any);

    let app = Router::new()
        .route("/news", get(handle_news))
        .route("/health", get(handle_health))
        .layer(cors)
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("포트 바인딩 실패");

    println!("[server] 리스닝 on 0.0.0.0:{port}");
    axum::serve(listener, app).await.expect("서버 오류");
}
