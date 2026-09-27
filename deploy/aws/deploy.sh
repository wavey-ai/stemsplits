#!/usr/bin/env bash
# Build and deploy the stems segment service.
#
#   deploy/aws/deploy.sh
#
# Region defaults to eu-north-1. Needs the weight bundle, built by
# tools/reference/export.py, and Docker with buildx for linux/arm64.
set -euo pipefail

# This deployment uses the default AWS credentials.
export AWS_PROFILE=default

cd "$(dirname "$0")/../.."

region="${STEMS_AWS_REGION:-eu-north-1}"
stack="${STEMS_AWS_STACK:-stems-prod}"
memory="${STEMS_AWS_MEMORY:-1769}"
reserved="${STEMS_AWS_RESERVED_CONCURRENCY:-0}"
key_file="deploy/aws/.api-key"
template="deploy/aws/template.yml"
skip_image_build="${STEMS_SKIP_IMAGE_BUILD:-false}"

if [[ ! -f tools/reference/out/bundle/bundle.json ]]; then
  echo "weight bundle missing; run tools/reference/.venv/bin/python tools/reference/export.py" >&2
  exit 1
fi

if [[ ! -f "$key_file" ]]; then
  umask 077
  openssl rand -hex 24 > "$key_file"
fi
api_key="$(tr -d '\r\n' < "$key_file")"

common=(--region "$region" --stack-name "$stack" --template-file "$template" \
  --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset)

# First pass creates ECR before an image URI exists.
if ! aws cloudformation describe-stacks --region "$region" --stack-name "$stack" >/dev/null 2>&1; then
  aws cloudformation deploy "${common[@]}" --parameter-overrides \
    DeployCompute=false ApiKeyValue="$api_key" MemorySize="$memory" ReservedConcurrency="$reserved"
fi

if [[ "$skip_image_build" == "true" ]]; then
  image_uri="$(aws cloudformation describe-stacks --region "$region" --stack-name "$stack" \
    --query 'Stacks[0].Parameters[?ParameterKey==`ImageUri`].ParameterValue' --output text)"
  deployment_token="$(aws cloudformation describe-stacks --region "$region" --stack-name "$stack" \
    --query 'Stacks[0].Parameters[?ParameterKey==`DeploymentToken`].ParameterValue' --output text)"
else
  repository="$(aws cloudformation describe-stacks --region "$region" --stack-name "$stack" \
    --query 'Stacks[0].Outputs[?OutputKey==`RepositoryUri`].OutputValue' --output text)"
  registry="${repository%%/*}"
  tag="$(date -u +%Y%m%d%H%M%S)"
  image="${repository}:${tag}"

  aws ecr get-login-password --region "$region" | docker login --username AWS --password-stdin "$registry"
  docker buildx build --platform linux/arm64 --provenance=false --load \
    -f deploy/aws/Dockerfile -t "$image" .
  docker push "$image"
  digest="$(aws ecr describe-images --region "$region" --repository-name "${repository#*/}" \
    --image-ids imageTag="$tag" --query 'imageDetails[0].imageDigest' --output text)"
  image_uri="${repository}@${digest}"
  deployment_token="$tag"
fi

aws cloudformation deploy "${common[@]}" --parameter-overrides \
  DeployCompute=true ImageUri="$image_uri" ApiKeyValue="$api_key" \
  MemorySize="$memory" ReservedConcurrency="$reserved" DeploymentToken="$deployment_token"

api_id="$(aws cloudformation describe-stacks --region "$region" --stack-name "$stack" \
  --query 'Stacks[0].Outputs[?OutputKey==`ApiId`].OutputValue' --output text)"
aws apigateway create-deployment --region "$region" --rest-api-id "$api_id" \
  --stage-name prod --description "production-${deployment_token}" >/dev/null

endpoint="$(aws cloudformation describe-stacks --region "$region" --stack-name "$stack" \
  --query 'Stacks[0].Outputs[?OutputKey==`Endpoint`].OutputValue' --output text)"
printf '%s\n' "endpoint: $endpoint" "key file: $key_file" "memory: ${memory} MB" \
  "reserved concurrency: $reserved"
