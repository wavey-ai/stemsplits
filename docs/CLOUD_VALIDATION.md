# Cloud validation

## Deployment

Validated on 24 September 2026.

- AWS profile: `default` only.
- Region: `eu-north-1`.
- Stack and function: `stems-prod`.
- Lambda: arm64, 1,769 MB, 120-second timeout.
- Image digest: `sha256:07a539d596995146e5d0750a38f785c700946c6e2f48653e3f68e42ecd6606c5`.
- Site deployment: `48cf041a-4ac6-416c-bc7e-2305870176ca`.

The browser calls the authenticated `yl-vin-stems` Worker through yl.vin.
The Worker calls API Gateway. Lambda streams the four f32 stems back.
The browser uses the shared chunk plan and overlap-add code to build four WAV files.

## Geometry

The production path retains 343,980 frames per segment at 44.1 kHz.
This is 7.8 seconds, with 25% overlap.
No shorter-window quality claim is made.

The extracted segment service matched the reference stems with a maximum
absolute sample error of `3.951e-6` on the reference fixture.
The streaming overlap test matched batch reconstruction exactly.
These checks do not replace a listening test or a music-quality benchmark.

## Checks

The following checks passed:

- Rust workspace tests, Lambda tests, and Clippy checks for the changed crates.
- All four model-fixture tests, with ignored tests explicitly enabled.
- Browser request ordering, cancellation, response size, format, and error tests.
- Compiled WASM duration and multi-segment reconstruction tests.
- Proxy authentication, origin, size, rate-limit, streaming, and no-cache tests.
- Isolated browser tests in Chromium and WebKit.
- Live Lambda inference through the signed-in yl.vin interface.
- The web performance guard and site-router tests.

The final live test used 16 seconds of synthetic stereo audio at 48 kHz.
The browser resampled it and submitted three overlapping model segments.
All four imported stems had the expected 16-second duration and a lossless library copy.
The test checked duplicate prevention, reload, deletion, and an empty restored library.
It removed its temporary identity and sessions after the run.

The final live test took 63.938 seconds in total. This includes inference,
transfers, library imports, screenshots, reload, and deletion checks.
It is an integration-test duration, not a throughput benchmark.

## Regression coverage

The browser tests cover a WASM32 duration-calculation overflow and concurrent
decoder initialization. The library test checks that deleting a source also
removes its stems from persistent storage.

Test commands are listed in the repository README under Cloud checks.

## Parallel requests and retries

The 24 September follow-up dispatches every segment in the existing plan concurrently.
The model weights, segment geometry, and reconstruction order remain unchanged.
Temporary OPFS files hold completed responses until reconstruction reads them.
Web Locks protect active scratch directories when the client removes abandoned files.

The request contract passed 33 Node tests across the stem and ECDC clients.
The tests include full fan-out, deadlines, response-body stalls, retry limits, throttling, cancellation, and cloud-to-local fallback.
The compiled proxy passed an isolated 104-request burst test.
Chromium and WebKit passed automatic retry, import, scratch cleanup, reload, and deletion checks.
A deterministic deletion test checks encoder shutdown before removal and restart for unrelated tracks.

One live 16-second stem test dispatched all three Lambda invocations at `11:26:45 UTC`.
It completed all four imports and library checks in 88.184 seconds.
Lambda reported invocation durations of 21.520, 65.567, and 68.107 seconds.
These observations confirm concurrent execution. They do not establish a throughput improvement or a cold-start distribution.

A live ECDC smoke test completed three parallel paired encodes and one decode in 2.885 seconds.
The decode returned 11,792 bytes. This is a smoke-test duration, not a benchmark.

The final site deployment for this follow-up is `61c26978-6d2e-45d2-9763-d4454c65d8f4`.
Its live 16-second test passed in 92.230 seconds, including import, reload, deletion, and scratch cleanup.
The test removed its temporary identity and sessions.
The AWS image is unchanged. AWS diagnostic calls used only the default profile.

## Regional deployment

The service runs in the same five regions as the ECDC service:

- `eu-west-1`
- `eu-north-1`
- `us-east-1`
- `us-west-2`
- `ap-southeast-1`

Each stack uses image digest `sha256:07a539d596995146e5d0750a38f785c700946c6e2f48653e3f68e42ecd6606c5`.
Each region has an account concurrency quota of 1,000 executions.

The authenticated Worker selects the nearest region from the request coordinates.
A retried request selects the next-nearest region and sends the same audio body.
The browser has three total attempts, so one request can use three regional endpoints.

A real model segment passed in all five regions. Each response contained 11,007,360 bytes.
All five responses had SHA-256 `ef1365885f166071defe0e66e57d6e3c82c73a769b62819351eb7b669325b0c0`.
The simultaneous cold smoke checks completed in 37.481 to 128.085 seconds.
These durations include client transfer time and do not form a regional performance benchmark.

Worker version `554e6bb3-930b-48d3-836f-589ffe388fe0` enabled regional routing.
Site version `14723a5b-7ff4-41fc-b64e-ef62e9ddd7fd` enabled regional retry metadata.

## Graviton2 kernels

Deployed on 3 October 2026 to the five regions.

- Image digest: `sha256:6cd30ccbe61ba74aff5aa42ac24811df519267f2f117cdd5600914052fb51ba0`.
- Lambda: arm64, 1,769 MB, 120-second timeout.
- Source: commit `ce12d0a`.

`deploy/aws/smoke.mjs` sends one synthetic segment. Before the deployment,
eu-north-1 returned SHA-256
`f10b6515e55956a9802a44e80255dd5721678111804c1a9fd96f6a873904b987`. After
the deployment, all five regions returned the same SHA-256.

Three warm requests went to each of two regions, in turn. eu-north-1 had
the new image and eu-west-1 had the previous image. The client times include
the transfer:

| Region | Image | Request 1 | Request 2 | Request 3 |
| --- | --- | ---: | ---: | ---: |
| eu-north-1 | New | 22.7 s | 22.8 s | 23.0 s |
| eu-west-1 | Previous | 64.9 s | 65.0 s | 66.2 s |

The browser WASM build gives the same output bits before and after these
changes. Its time for one segment in Node on an Apple M1 is 13 to 14 s
before and after.
