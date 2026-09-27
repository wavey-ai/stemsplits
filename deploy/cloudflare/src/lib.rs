//! Authenticated segment requests for yl.vin. The AWS key stays in a Worker secret.
use futures_util::StreamExt;
use serde_json::{json, Value};
use worker::*;

const OPUS_INPUT_BYTES: usize = 512 * 1024;
const OPUS_MIN_INPUT_BYTES: usize = 1024;
const REGIONS: [(&str, f32, f32); 5] = [
    ("eu-west-1", 53.35, -6.26),
    ("eu-north-1", 59.33, 18.07),
    ("us-east-1", 38.95, -77.45),
    ("us-west-2", 45.84, -119.70),
    ("ap-southeast-1", 1.35, 103.82),
];

fn failure(status: u16, message: &str) -> Result<Response> {
    Ok(Response::from_json(&json!({"error": message}))?.with_status(status))
}

fn distance(latitude: f32, longitude: f32, target_latitude: f32, target_longitude: f32) -> f32 {
    let latitude_scale = latitude.to_radians().cos();
    let north = latitude - target_latitude;
    let east = (longitude - target_longitude) * latitude_scale;
    north * north + east * east
}

fn nearest_region(latitude: f32, longitude: f32, attempt: usize) -> &'static str {
    let mut regions = REGIONS;
    regions.sort_by(|left, right| {
        distance(latitude, longitude, left.1, left.2)
            .total_cmp(&distance(latitude, longitude, right.1, right.2))
    });
    regions[attempt.min(regions.len() - 1)].0
}

fn selected_region(request: &Request) -> &'static str {
    let attempt = request
        .headers()
        .get("X-Stems-Attempt")
        .ok()
        .flatten()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    if let Some((latitude, longitude)) = request.cf().and_then(|cf| cf.coordinates()) {
        return nearest_region(latitude, longitude, attempt);
    }
    let continent = request.cf().and_then(|cf| cf.continent());
    let order = match continent.as_deref() {
        Some("NA" | "SA") => [
            "us-east-1",
            "us-west-2",
            "eu-west-1",
            "eu-north-1",
            "ap-southeast-1",
        ],
        Some("AS" | "OC") => [
            "ap-southeast-1",
            "us-west-2",
            "eu-west-1",
            "us-east-1",
            "eu-north-1",
        ],
        Some("AF") => [
            "eu-west-1",
            "eu-north-1",
            "ap-southeast-1",
            "us-east-1",
            "us-west-2",
        ],
        _ => [
            "eu-north-1",
            "eu-west-1",
            "us-east-1",
            "us-west-2",
            "ap-southeast-1",
        ],
    };
    order[attempt.min(order.len() - 1)]
}

fn endpoint(env: &Env, region: &str) -> Result<String> {
    let name = match region {
        "eu-west-1" => "STEMS_ENDPOINT_EU_WEST_1",
        "eu-north-1" => "STEMS_ENDPOINT_EU_NORTH_1",
        "us-east-1" => "STEMS_ENDPOINT_US_EAST_1",
        "us-west-2" => "STEMS_ENDPOINT_US_WEST_2",
        "ap-southeast-1" => "STEMS_ENDPOINT_AP_SOUTHEAST_1",
        _ => return Err(Error::RustError("Unknown stem region.".into())),
    };
    Ok(env.var(name)?.to_string())
}

async fn session_key(request: &Request, env: &Env) -> Result<Option<String>> {
    let origin = env.var("PUBLIC_ORIGIN")?.to_string();
    let headers = Headers::new();
    headers.set("Accept", "application/json")?;
    headers.set("Origin", &origin)?;
    if let Some(cookie) = request.headers().get("Cookie")? {
        headers.set("Cookie", &cookie)?;
    }
    let mut init = RequestInit::new();
    init.with_method(Method::Get).with_headers(headers);
    let mut response = env
        .service("IDENTITY")?
        .fetch_request(Request::new_with_init(
            &format!("{origin}/id/session"),
            &init,
        )?)
        .await?;
    if response.status_code() != 200 {
        return Ok(None);
    }
    let session: Value = response.json().await?;
    let sub = session["user"]["sub"].as_str().unwrap_or_default();
    Ok((session["authenticated"] == true
        && session["user"]["email_verified"] == true
        && !sub.is_empty())
    .then(|| sub.to_owned()))
}

async fn dispatch(mut request: Request, env: Env) -> Result<Response> {
    let path = request.path();
    if path == "/api/stems/health" && request.method() == Method::Get {
        return Response::from_json(&json!({
            "ok": true,
            "format": "soundkit-v2-opus-192",
            "regions": REGIONS.map(|region| region.0),
        }));
    }
    if !matches!(
        (request.method(), path.as_str()),
        (Method::Get, "/api/stems/status") | (Method::Post, "/api/stems/separate")
    ) {
        return failure(404, "Unknown stem request.");
    }
    let origin = env.var("PUBLIC_ORIGIN")?.to_string();
    if request
        .headers()
        .get("Origin")?
        .is_some_and(|value| value != origin)
    {
        return failure(403, "Request origin is not allowed.");
    }
    let Some(user) = session_key(&request, &env).await? else {
        return failure(401, "Sign in to split this track.");
    };
    if request.method() == Method::Get {
        return Response::from_json(&json!({"ok": true, "region": selected_region(&request)}));
    }
    if request.headers().get("Content-Type")?.as_deref() != Some("application/octet-stream")
        || request.headers().get("X-Stems-Format")?.as_deref() != Some("soundkit-v2-opus-192")
    {
        return failure(415, "Invalid audio format.");
    }
    let input_limit = OPUS_INPUT_BYTES;
    if !env
        .rate_limiter("STEMS_LIMITER")?
        .limit(user)
        .await?
        .success
    {
        let response = failure(429, "Please wait a minute before splitting more audio.")?;
        response.headers().set("Retry-After", "60")?;
        return Ok(response);
    }
    let body = if let Some(length) = request.headers().get("Content-Length")? {
        let length = length.parse::<usize>().unwrap_or(0);
        if length > input_limit {
            return failure(413, "Audio segment is too large.");
        }
        if length < OPUS_MIN_INPUT_BYTES {
            return failure(400, "Audio segment is incomplete.");
        }
        // Forward fixed-length uploads without retaining one PCM buffer per active request.
        // Lambda validates the received body before model inference.
        request.inner().body().map(Into::into)
    } else {
        let mut bytes = Vec::with_capacity(input_limit);
        let mut stream = request.stream()?;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            if bytes.len() + chunk.len() > input_limit {
                return failure(413, "Audio segment is too large.");
            }
            bytes.extend_from_slice(&chunk);
        }
        if bytes.len() < OPUS_MIN_INPUT_BYTES {
            return failure(400, "Audio segment is incomplete.");
        }
        Some(js_sys::Uint8Array::from(bytes.as_slice()).into())
    };
    let headers = Headers::new();
    headers.set("Content-Type", "application/octet-stream")?;
    headers.set("Accept", "application/octet-stream")?;
    headers.set("X-Stems-Format", "soundkit-v2-opus-192")?;
    headers.set("X-Api-Key", &env.secret("STEMS_API_KEY")?.to_string())?;
    let mut init = RequestInit::new();
    init.with_method(Method::Post)
        .with_headers(headers)
        .with_body(body);
    let region = selected_region(&request);
    let upstream = Request::new_with_init(&endpoint(&env, region)?, &init)?;
    drop(init);
    let response = Fetch::Request(upstream).send().await?;
    if response.status_code() != 200 {
        let failure = failure(
            if response.status_code() == 429 {
                429
            } else {
                502
            },
            "Could not split this segment. Please try again.",
        )?;
        if let Some(delay) = response.headers().get("Retry-After")? {
            failure.headers().set("Retry-After", &delay)?;
        }
        return Ok(failure);
    }
    if response.headers().get("X-Stems-Format")?.as_deref() != Some("soundkit-v2-opus-192") {
        return failure(502, "The stem response has an unsupported format.");
    }
    let headers = response.headers().clone();
    headers.set("X-Stems-Region", region)?;
    Ok(response.with_headers(headers))
}

#[event(fetch)]
pub async fn main(request: Request, env: Env, _: Context) -> Result<Response> {
    let response = match dispatch(request, env).await {
        Ok(response) => response,
        Err(_) => failure(502, "Stem splitting is unavailable. Please try again.")?,
    };
    let headers = response.headers().clone();
    headers.set("Cache-Control", "no-store")?;
    Ok(response.with_headers(headers))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geography_selects_the_nearest_region_and_retries_elsewhere() {
        assert_eq!(nearest_region(51.51, -0.13, 0), "eu-west-1");
        assert_eq!(nearest_region(59.33, 18.07, 0), "eu-north-1");
        assert_eq!(nearest_region(40.71, -74.01, 0), "us-east-1");
        assert_eq!(nearest_region(37.77, -122.42, 0), "us-west-2");
        assert_eq!(nearest_region(1.35, 103.82, 0), "ap-southeast-1");
        assert_ne!(
            nearest_region(51.51, -0.13, 0),
            nearest_region(51.51, -0.13, 1)
        );
    }
}
