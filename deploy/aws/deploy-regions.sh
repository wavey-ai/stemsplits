#!/usr/bin/env bash
# Deploy the same stems image and API key to the production regions.
set -euo pipefail

export AWS_PROFILE=default

cd "$(dirname "$0")/../.."

regions=(eu-west-1 eu-north-1 us-east-1 us-west-2 ap-southeast-1)
source_region="${STEMS_SOURCE_REGION:-eu-north-1}"
stack="${STEMS_AWS_STACK:-stems-prod}"
memory="${STEMS_AWS_MEMORY:-1769}"
reserved="${STEMS_AWS_RESERVED_CONCURRENCY:-0}"
template="deploy/aws/template.yml"
key_file="deploy/aws/.api-key"

test -s "$key_file"
api_key="$(tr -d '\r\n' < "$key_file")"
source_repository="$(aws cloudformation describe-stacks --region "$source_region" --stack-name "$stack" \
  --query 'Stacks[0].Outputs[?OutputKey==`RepositoryUri`].OutputValue' --output text)"
source_image="$(aws cloudformation describe-stacks --region "$source_region" --stack-name "$stack" \
  --query 'Stacks[0].Parameters[?ParameterKey==`ImageUri`].ParameterValue' --output text)"
source_digest="${source_image##*@}"
tag="multi-$(date -u +%Y%m%d%H%M%S)"

aws ecr get-login-password --region "$source_region" | \
  docker login --username AWS --password-stdin "${source_repository%%/*}"
docker pull "${source_repository}@${source_digest}"

for region in "${regions[@]}"; do
  common=(--region "$region" --stack-name "$stack" --template-file "$template" \
    --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset)
  if ! aws cloudformation describe-stacks --region "$region" --stack-name "$stack" >/dev/null 2>&1; then
    aws cloudformation deploy "${common[@]}" --parameter-overrides \
      DeployCompute=false ApiKeyValue="$api_key" MemorySize="$memory" ReservedConcurrency="$reserved"
  fi

  repository="$(aws cloudformation describe-stacks --region "$region" --stack-name "$stack" \
    --query 'Stacks[0].Outputs[?OutputKey==`RepositoryUri`].OutputValue' --output text)"
  image="${repository}:${tag}"
  aws ecr get-login-password --region "$region" | \
    docker login --username AWS --password-stdin "${repository%%/*}"
  docker tag "${source_repository}@${source_digest}" "$image"
  docker push "$image"
  digest="$(aws ecr describe-images --region "$region" --repository-name "${repository#*/}" \
    --image-ids imageTag="$tag" --query 'imageDetails[0].imageDigest' --output text)"
  if [[ "$digest" != "$source_digest" ]]; then
    echo "image digest differs in $region" >&2
    exit 1
  fi

  aws cloudformation deploy "${common[@]}" --parameter-overrides \
    DeployCompute=true ImageUri="${repository}@${digest}" ApiKeyValue="$api_key" \
    MemorySize="$memory" ReservedConcurrency="$reserved" DeploymentToken="$tag"
  api_id="$(aws cloudformation describe-stacks --region "$region" --stack-name "$stack" \
    --query 'Stacks[0].Outputs[?OutputKey==`ApiId`].OutputValue' --output text)"
  aws apigateway create-deployment --region "$region" --rest-api-id "$api_id" \
    --stage-name prod --description "production-${tag}" >/dev/null
done

for region in "${regions[@]}"; do
  endpoint="$(aws cloudformation describe-stacks --region "$region" --stack-name "$stack" \
    --query 'Stacks[0].Outputs[?OutputKey==`Endpoint`].OutputValue' --output text)"
  printf '%s %s/separate\n' "$region" "$endpoint"
done
