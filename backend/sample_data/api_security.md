# API Security & Authentication Protocol

## 1. Authentication Standards
All external requests to the NovoMind microservice cluster must be authenticated using JSON Web Tokens (JWT) signed with RS256 asymmetric keys.

## 2. Rate Limiting Policy
- **Free Tier**: 60 requests per minute per IP address.
- **Enterprise Tier**: 1,200 requests per minute with dedicated vector index throughput.

## 3. Vector Database Encryption
All persistent vector storage indexes (ChromaDB / FAISS snapshots) are encrypted at rest using AES-256-GCM. Document metadata is cleansed of PII (Personally Identifiable Information) before embedding generation.
