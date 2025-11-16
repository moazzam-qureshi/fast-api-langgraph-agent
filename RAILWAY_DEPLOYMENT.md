# Railway Deployment Guide

This guide will help you deploy the FastAPI LangGraph Agent to Railway.

## Prerequisites

1. A Railway account (sign up at https://railway.app)
2. Railway CLI installed (optional but recommended)
3. Your environment variables ready (OpenAI API key, etc.)

## Deployment Steps

### 1. Initial Setup on Railway

1. Go to [Railway Dashboard](https://railway.app/dashboard)
2. Click "New Project"
3. Choose "Deploy from GitHub repo"
4. Select your repository: `moazzam-qureshi/fast-api-langgraph-agent`
5. Railway will automatically detect the Docker setup

### 2. Add Required Services

Railway will need to provision the following services:

#### PostgreSQL with PGVector
1. In your Railway project, click "New Service"
2. Choose "Database" → "Add PostgreSQL"
3. After PostgreSQL is created, go to its settings
4. In the "Docker Image" section, change the image to: `pgvector/pgvector:pg16`
5. Deploy the changes

#### Redis
1. Click "New Service"
2. Choose "Database" → "Add Redis"

#### Application Services
Railway should automatically detect and create services for:
- `app` (FastAPI application)
- `celery_worker` (Background task processor)
- `minio` (Object storage)

### 3. Configure Environment Variables

For each service, add the required environment variables:

#### For the `app` service:
```env
# Required
OPENAI_API_KEY=your-openai-api-key
JWT_SECRET_KEY=generate-a-secure-random-key

# MinIO credentials
MINIO_ROOT_USER=your-minio-admin
MINIO_ROOT_PASSWORD=your-secure-minio-password

# Optional LangChain tracing
LANGCHAIN_API_KEY=your-langchain-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=fastapi-langgraph
```

#### For the `celery_worker` service:
Add the same environment variables as the `app` service.

### 4. Configure Service Dependencies

1. Make sure `app` and `celery_worker` depend on:
   - PostgreSQL
   - Redis
   - MinIO

2. Railway will automatically inject connection strings:
   - `DATABASE_URL` for PostgreSQL
   - `REDIS_URL` for Redis

### 5. Deploy

1. Railway will automatically deploy when you push to your GitHub repository
2. Monitor the deployment logs for each service
3. Wait for all services to be healthy

### 6. Post-Deployment Setup

#### Create MinIO Bucket
1. Access MinIO console through Railway's service URL (port 9001)
2. Login with your MINIO_ROOT_USER and MINIO_ROOT_PASSWORD
3. Create a bucket named "documents"

#### Verify Health
1. Access your app's URL provided by Railway
2. Check health endpoints:
   - `https://your-app.railway.app/health` - Basic health check
   - `https://your-app.railway.app/health/detailed` - Comprehensive service check
   - `https://your-app.railway.app/docs` - API documentation

## Environment Variables Reference

### Required Variables
- `OPENAI_API_KEY` - Your OpenAI API key for embeddings and LLM
- `JWT_SECRET_KEY` - Secure random key for JWT tokens (generate with `openssl rand -hex 32`)

### Auto-Injected by Railway
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `PORT` - Application port

### MinIO Configuration
- `MINIO_ROOT_USER` - MinIO admin username
- `MINIO_ROOT_PASSWORD` - MinIO admin password

### Optional Variables
- `LANGCHAIN_API_KEY` - For LangChain tracing
- `LANGCHAIN_TRACING_V2` - Enable tracing (true/false)
- `LANGCHAIN_PROJECT` - Project name for tracing

## Troubleshooting

### PGVector Extension Issues
If you get pgvector errors:
1. Ensure PostgreSQL service is using `pgvector/pgvector:pg16` image
2. The app automatically creates the extension on startup

### MinIO Connection Issues
1. Verify MinIO service is running
2. Check MINIO_ENDPOINT is set to `minio:9000` (internal service name)
3. Ensure bucket "documents" exists

### Celery Worker Issues
1. Check Redis service is healthy
2. Verify CELERY_BROKER_URL and CELERY_RESULT_BACKEND are set correctly
3. Monitor celery worker logs for errors

### Health Check Failures
Use the `/health/detailed` endpoint to identify which service is failing:
```bash
curl https://your-app.railway.app/health/detailed
```

## Performance Optimization

1. **Scaling**: Increase replicas for `app` and `celery_worker` services as needed
2. **Database**: Consider upgrading PostgreSQL instance for production loads
3. **Redis**: Monitor memory usage and upgrade if needed
4. **MinIO**: For large file storage, consider external S3-compatible storage

## Security Considerations

1. Always use strong, unique passwords for all services
2. Rotate JWT_SECRET_KEY periodically
3. Use Railway's private networking for inter-service communication
4. Enable HTTPS (Railway provides this automatically)
5. Regularly update dependencies

## Monitoring

1. Use Railway's built-in logs and metrics
2. Monitor the `/health/detailed` endpoint
3. Set up alerts for service failures
4. Track OpenAI API usage and costs