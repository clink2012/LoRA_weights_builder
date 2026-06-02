FROM node:22-alpine AS build

WORKDIR /app/Database/UI

COPY Database/UI/package*.json ./
RUN npm ci

COPY Database/UI/ ./
RUN npm run build

FROM nginx:1.27-alpine

COPY deploy/nibbler/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/Database/UI/dist /usr/share/nginx/html

EXPOSE 80
