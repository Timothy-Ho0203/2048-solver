# Docker Deployment Guide

This guide covers containerizing and deploying the 2048 AI project using Docker.

## 📁 Project Structure

```
2048-Solver/
├── backend/
│   ├── Dockerfile              # Development Docker image
│   ├── Dockerfile.prod         # Production-optimized image
│   ├── .dockerignore
│   ├── requirements.txt
│   └── api.py
├── frontend/
│   ├── Dockerfile              # Production Nginx image
│   ├── package.json
│   └── ...
└── docker-compose.yml          # Multi-service orchestration
```

## 🐳 Quick Start with Docker Compose

### Run Both Services (Recommended for Local Development)

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Access:**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 🔨 Backend Only (Manual Docker)

### Development Build

```bash
cd backend

# Build the image
docker build -t 2048-backend:dev .

# Run the container
docker run -p 8000:8000 --name 2048-backend 2048-backend:dev

# Run with environment variables
docker run -p 8000:8000 \
  -e PORT=8000 \
  -e HOST=0.0.0.0 \
  --name 2048-backend \
  2048-backend:dev

# Run with volume mount (hot reload)
docker run -p 8000:8000 \
  -v $(pwd):/app \
  -v /app/venv \
  --name 2048-backend \
  2048-backend:dev
```

### Production Build (Optimized)

```bash
cd backend

# Build production image
docker build -f Dockerfile.prod -t 2048-backend:prod .

# Run production container
docker run -p 8000:8000 \
  --name 2048-backend-prod \
  --restart unless-stopped \
  2048-backend:prod
```

### Backend Commands

```bash
# View logs
docker logs -f 2048-backend

# Execute commands inside container
docker exec -it 2048-backend bash

# Stop container
docker stop 2048-backend

# Remove container
docker rm 2048-backend

# Remove image
docker rmi 2048-backend:dev
```

---

## 🎨 Frontend (Manual Docker)

### Build and Run

```bash
cd frontend

# Build the image
docker build -t 2048-frontend:prod .

# Run the container
docker run -p 80:80 \
  --name 2048-frontend \
  2048-frontend:prod

# Or map to different port
docker run -p 5173:80 \
  --name 2048-frontend \
  2048-frontend:prod
```

---

## 🚀 Deployment Scenarios

### Scenario 1: Deploy to Cloud Run (Google Cloud)

```bash
# Build and tag for GCR
cd backend
docker build -t gcr.io/YOUR_PROJECT/2048-backend:latest .

# Push to Google Container Registry
docker push gcr.io/YOUR_PROJECT/2048-backend:latest

# Deploy to Cloud Run
gcloud run deploy 2048-backend \
  --image gcr.io/YOUR_PROJECT/2048-backend:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000
```

### Scenario 2: Deploy to AWS ECS

```bash
# Build and tag for ECR
cd backend
docker build -t 2048-backend:latest .

# Tag for ECR
docker tag 2048-backend:latest \
  YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/2048-backend:latest

# Push to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/2048-backend:latest

# Create ECS task definition and service via AWS Console or CLI
```

### Scenario 3: Deploy to Render/Railway/Fly.io

These platforms auto-detect Dockerfiles. Just:
1. Push code to GitHub
2. Connect repository to platform
3. Platform auto-builds and deploys

**For Railway:**
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

**For Fly.io:**
```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Launch app
cd backend
fly launch

# Deploy
fly deploy
```

### Scenario 4: Deploy to DigitalOcean App Platform

1. Push to GitHub
2. Create new App in DigitalOcean
3. Point to your repository
4. Select `backend` directory
5. Auto-detects Dockerfile
6. Deploy

---

## 🔧 Docker Optimization Tips

### 1. Reduce Image Size

```dockerfile
# Use slim base images
FROM python:3.11-slim  # ~150MB vs python:3.11 (~900MB)

# Multi-stage builds
# Separate build dependencies from runtime
```

### 2. Layer Caching

```dockerfile
# Copy requirements BEFORE code
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .  # This layer only rebuilds when code changes
```

### 3. .dockerignore

Always exclude:
- `venv/`, `node_modules/`
- `__pycache__/`, `*.pyc`
- `.git/`, `.vscode/`
- Large model files (download at runtime if needed)

### 4. Security

```dockerfile
# Run as non-root user
RUN useradd -m appuser
USER appuser

# Use specific versions
FROM python:3.11-slim  # Not 'latest'
```

---

## 📊 Resource Management

### Set Resource Limits

```bash
# Limit memory and CPU
docker run -p 8000:8000 \
  --memory="2g" \
  --cpus="1.5" \
  2048-backend:dev
```

### Docker Compose with Limits

```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```

---

## 🐛 Troubleshooting

### Check Container Logs

```bash
docker logs 2048-backend
docker logs -f 2048-backend  # Follow logs
```

### Inspect Container

```bash
docker inspect 2048-backend
docker stats 2048-backend  # Resource usage
```

### Debug Inside Container

```bash
docker exec -it 2048-backend bash
docker exec -it 2048-backend python -c "import torch; print(torch.__version__)"
```

### Health Checks

```bash
# Manual health check
curl http://localhost:8000/health

# Docker health status
docker ps  # Shows health status
```

### Common Issues

**Issue: Port already in use**
```bash
# Find process using port
lsof -ti:8000
# Kill process
kill -9 $(lsof -ti:8000)
```

**Issue: Container exits immediately**
```bash
# Check logs
docker logs 2048-backend
# Run interactively
docker run -it 2048-backend:dev bash
```

**Issue: Model files too large**
```bash
# Download models at runtime instead of including in image
# Or use .dockerignore and mount volume
docker run -v $(pwd)/models:/app/models 2048-backend:dev
```

---

## 📈 Production Checklist

- [ ] Use multi-stage builds (`Dockerfile.prod`)
- [ ] Set resource limits (memory, CPU)
- [ ] Enable health checks
- [ ] Run as non-root user
- [ ] Use environment variables for config
- [ ] Set up logging and monitoring
- [ ] Configure CORS properly
- [ ] Use HTTPS (TLS termination at load balancer)
- [ ] Set up CI/CD pipeline
- [ ] Implement secrets management
- [ ] Configure auto-scaling

---

## 🔒 Security Best Practices

1. **Never include secrets in Dockerfile**
   ```bash
   # Use environment variables
   docker run -e API_KEY=$API_KEY 2048-backend
   ```

2. **Scan images for vulnerabilities**
   ```bash
   docker scan 2048-backend:prod
   ```

3. **Use official base images**
   ```dockerfile
   FROM python:3.11-slim  # Official Python image
   ```

4. **Keep images updated**
   ```bash
   docker pull python:3.11-slim
   docker build --no-cache -t 2048-backend:prod .
   ```

---

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Best Practices for Writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Multi-stage Builds](https://docs.docker.com/build/building/multi-stage/)
