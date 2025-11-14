# Multi-stage build for Next.js frontend
# Build-time variables (can be overridden with --build-arg):
# - NODE_ENV: Build environment (default: production)
FROM node:18-alpine AS builder

# Add labels for better tracking
LABEL maintainer="Fund Analysis" \
      version="1.0" \
      description="Fund Analysis Frontend Application"

# Set working directory
WORKDIR /app

# Copy package files first for better caching
COPY package*.json ./

# Install all dependencies (including dev dependencies for build)
RUN npm ci && npm cache clean --force

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM node:18-alpine

# Add labels
LABEL maintainer="Fund Analysis" \
      version="1.0" \
      description="Fund Analysis Frontend Application"

# Install curl for health checks
RUN apk add --no-cache curl

# Create non-root user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install only production dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy built application from builder stage
COPY --from=builder --chown=nextjs:nodejs /app/.next ./.next
COPY --from=builder --chown=nextjs:nodejs /app/next.config.js ./
# Note: public directory is optional in Next.js - not present in this project

# Switch to non-root user
USER nextjs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/api/health || exit 1

# Start the application
CMD ["npm", "start"]