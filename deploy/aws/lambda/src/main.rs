//! One HTDemucs segment per request. Responses use Lambda response streaming.
use lambda_http::{run_with_streaming_response, service_fn, Body, Error, Request, Response};
use std::{path::Path, sync::OnceLock};
use stemsplits_htdemucs::{model::HtDemucs, separate::separate_segment};
use stemsplits_model::Weights;
use stemsplits_stft::{Geometry, Stft};

const SEGMENT: usize = Geometry::CONTRACT.segment;
static MODEL: OnceLock<Result<HtDemucs, String>> = OnceLock::new();

fn model() -> Result<&'static HtDemucs, Error> {
    MODEL
        .get_or_init(|| {
            let directory = std::env::var("BUNDLE_DIR").unwrap_or_else(|_| "/opt/bundle".into());
            Weights::open(Path::new(&directory))
                .and_then(|weights| HtDemucs::load(&weights))
                .map_err(|error| format!("{error:#}"))
        })
        .as_ref()
        .map_err(|error| error.clone().into())
}

fn response(status: u16, body: Body, content_type: &str) -> Result<Response<Body>, Error> {
    Ok(Response::builder()
        .status(status)
        .header("content-type", content_type)
        .header("cache-control", "no-store")
        .body(body)?)
}

fn failure(status: u16, message: &str) -> Result<Response<Body>, Error> {
    response(status, Body::Text(message.into()), "text/plain")
}

async fn handler(request: Request) -> Result<Response<Body>, Error> {
    if request.method() != "POST" {
        return failure(405, "Method not allowed");
    }
    let expected = std::env::var("API_KEY").unwrap_or_default();
    let provided = request
        .headers()
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    if expected.is_empty() || provided != expected {
        return failure(403, "Forbidden");
    }
    let format = request
        .headers()
        .get("x-stems-format")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    if format != stemsplits_transport::FORMAT {
        return failure(400, "Expected soundkit-v2-opus-192");
    }
    let Body::Binary(body) = request.body() else {
        return failure(400, "Expected application/octet-stream");
    };
    let [left, right] = match stemsplits_transport::decode(body, 1, SEGMENT) {
        Ok(mut streams) => streams.remove(0),
        Err(_) => return failure(400, "Invalid Opus segment"),
    };
    let mut stft = Stft::new(Geometry::CONTRACT);
    let stems = separate_segment(model()?, &left, &right, &mut stft);
    let bytes = match stemsplits_transport::encode(&stems) {
        Ok(bytes) => bytes,
        Err(_) => return failure(500, "Could not encode model output"),
    };
    let mut output = response(200, Body::Binary(bytes), "application/octet-stream")?;
    output.headers_mut().insert("x-stems-format", format.parse()?);
    Ok(output)
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    run_with_streaming_response(service_fn(handler)).await
}
