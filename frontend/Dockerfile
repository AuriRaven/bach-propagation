FROM node:20-alpine

WORKDIR /app

# Copiar archivos de dependencias
COPY package.json pnpm-lock.yaml* ./

# Instalar pnpm y dependencias (v0 suele usar pnpm)
RUN corepack enable && corepack prepare pnpm@latest --activate
RUN pnpm install

# Copiar el resto del código
COPY . .

EXPOSE 3000

CMD ["pnpm", "run", "dev"]