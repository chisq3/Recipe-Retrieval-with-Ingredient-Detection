// Thin client for the demo backend (proxied via vite.config.js to :8000).

export async function detectIngredients(file) {
  const form = new FormData()
  form.append('image', file)
  const res = await fetch('/detect', { method: 'POST', body: form })
  if (!res.ok) throw new Error(`/detect failed (${res.status})`)
  const data = await res.json()
  return data.detections || []
}

export async function recommend(query, ingredients) {
  const res = await fetch('/recommend', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, ingredients }),
  })
  if (!res.ok) throw new Error(`/recommend failed (${res.status})`)
  return res.json()
}
