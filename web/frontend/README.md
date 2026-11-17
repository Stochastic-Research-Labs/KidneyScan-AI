# KidneyScan AI frontend (drop-in for `web/frontend`)

Copy `app/`, `components/` and `lib/` into `web/frontend` (or into `web/frontend/src/` if your project uses a `src` folder; all imports are relative, so either works). Replace the existing files with the same names.

```bash
cd web/frontend
npm i three nifti-reader-js
npm i -D @types/three
npm run dev
```

Needs Next.js 14 or 15 (App Router) and TypeScript. No Tailwind or other config needed.

Optional: point at a different backend with `NEXT_PUBLIC_API_BASE=https://...` in `.env.local`.
Default: https://m-wajid-t-v--kidneyscan-ai-serve.us-east.modal.run

## If the real model mode shows an error about the response
Everything that depends on the exact shape of `POST /api/predict`'s response lives in `lib/api.ts`
(`parseResponse`, `resolveMask`, `readScores`). Compare it with `API_CONTRACT.md` and adjust there only.
