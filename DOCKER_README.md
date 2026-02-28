# SwasthyaSetu - Docker Deployment Guide

This guide explains how to build and run the SwasthyaSetu FastAPI application using Docker and Docker Compose.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (version 20.10 or later)
- [Docker Compose](https://docs.docker.com/compose/install/) (version 1.29 or later)
- A valid `GOOGLE_API_KEY` from Google AI Studio

## Quick Start

### 1. Environment Setup

Create a `.env` file in the project root with your API key:

```bash
# Copy the example environment file
cp .env.example .env  # If available, or create manually

# Edit .env and add your Google API key
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

**Required Environment Variable:**
- `GOOGLE_API_KEY` - Your Google Generative AI API key

**Optional Environment Variables:**
- `DIAGNOSE_PER_IP_PER_MINUTE` - Rate limit per IP per minute (default: 4)
- `DIAGNOSE_PER_IP_PER_HOUR` - Rate limit per IP per hour (default: 30)
- `DIAGNOSE_GLOBAL_PER_MINUTE` - Global rate limit per minute (default: 20)
- `DIAGNOSE_GLOBAL_PER_HOUR` - Global rate limit per hour (default: 240)

### 2. Build and Run

```bash
# Build and start the containers in detached mode
docker-compose up -d

# View logs
docker-compose logs -f swasthyasetu

# Stop the containers
docker-compose down
```

### 3. Access the Application

Once running, the application is available at:
- **Web UI**: http://localhost:8000
- **API Health Check**: http://localhost:8000/health
- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **API Documentation (ReDoc)**: http://localhost:8000/redoc

## Available Commands

### Building

```bash
# Build the image
docker-compose build

# Build with no cache (clean build)
docker-compose build --no-cache
```

### Running

```bash
# Start services
docker-compose up -d

# Start with verbose logging
docker-compose up

# Restart a specific service
docker-compose restart swasthyasetu
```

### Stopping

```bash
# Stop services (containers remain)
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop and remove containers and volumes
docker-compose down -v
```

### Monitoring

```bash
# View logs
docker-compose logs swasthyasetu

# Follow logs (real-time)
docker-compose logs -f swasthyasetu

# Check service status
docker-compose ps
```

### Maintenance

```bash
# Execute commands in the container
docker-compose exec swasthyasetu /bin/bash

# Check disk usage
docker system df

# Clean up unused resources
docker system prune
```

## Directory Structure

The following directories are mounted as volumes:

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./static` | `/app/static` | Static assets (CSS, JS, images) |
| `./templates` | `/app/templates` | Jinja2 HTML templates |
| `./pubmed_data` | `/app/pubmed_data` | PubMed data for RAG (create if not exists) |
| `swasthyasetu-data` (named volume) | `/app/data` | Persistent application data |

## Production Deployment

### Resource Limits

Uncomment the resource limits in `docker-compose.yml` for production:

```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
    reservations:
      cpus: '1.0'
      memory: 2G
```

### Security Considerations

1. **Never commit the `.env` file** to version control
2. Use a non-root user in the Dockerfile (uncomment the USER directive)
3. Consider using Docker secrets for sensitive data in production
4. Place the container behind a reverse proxy (nginx/traefik) with HTTPS

### Health Checks

The container includes health checks that monitor the `/health` endpoint:
- **Interval**: 30 seconds
- **Timeout**: 10 seconds
- **Retries**: 3
- **Start Period**: 60 seconds

View health status:
```bash
docker-compose ps
```

## Troubleshooting

### Container fails to start

1. Check logs:
   ```bash
   docker-compose logs swasthyasetu
   ```

2. Verify environment variables:
   ```bash
   docker-compose exec swasthyasetu env | grep GOOGLE
   ```

3. Ensure `pubmed_data` directory exists:
   ```bash
   mkdir -p pubmed_data
   ```

### Port already in use

If port 8000 is already in use, change the port mapping in `docker-compose.yml`:
```yaml
ports:
  - "8080:8000"  # Maps host port 8080 to container port 8000
```

### Out of memory errors

Increase the memory limit or reduce the number of uvicorn workers:
```yaml
environment:
  - UVICORN_WORKERS=2  # Reduce from default 4
```

## Docker Compose Configuration Reference

| Setting | Value | Description |
|---------|-------|-------------|
| Service Name | `swasthyasetu` | Main application service |
| Build Context | `.` | Current directory |
| Exposed Port | `8000:8000` | Maps container port 8000 to host |
| Restart Policy | `unless-stopped` | Restarts on failure unless manually stopped |
| Health Check | `/health` endpoint | HTTP GET request every 30s |

## Uninstall

To completely remove all containers, volumes, and images:

```bash
# Stop and remove everything
docker-compose down -v --rmi all

# Remove named volume manually (if needed)
docker volume rm swasthyasetu_swasthyasetu-data
```

## Support

For issues or questions:
1. Check the logs: `docker-compose logs swasthyasetu`
2. Verify your API key is correctly set in `.env`
3. Ensure all required directories exist
