# --- build stage -----------------------------------------------------
FROM node:20-slim AS build
WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

COPY . .
# VITE_API_BASE / VITE_WS_BASE default to "" / "/api/sim" (relative), which
# is correct for this single-origin nginx setup -- override at build time
# only if the frontend is served from a different origin than the API.
ARG VITE_API_BASE=/api/sim
ARG VITE_WS_BASE=
ENV VITE_API_BASE=$VITE_API_BASE
ENV VITE_WS_BASE=$VITE_WS_BASE
RUN npm run build

# --- serve stage -------------------------------------------------------
FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD wget -q -O- http://127.0.0.1:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
