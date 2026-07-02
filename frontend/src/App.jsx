import { useEffect, useRef, useState } from 'react'
import { detectIngredients, recommend } from './api'
import { AssistantText, UserTurn, Loading, ReviewChips, RecipeCard, NoSafeCard, OutOfScopeCard, Composer } from './components.jsx'

const GREETING =
  'Hi! I help you find recipes from the ingredients you have. Type a request (e.g. "vegetarian pho, no oven") and/or attach a photo of your ingredients.'
const BG = 'https://lh3.googleusercontent.com/aida-public/AB6AXuBba_WHKT6_S3Rk3d4haqBfsfcluat4neetc5859gUeoqgqIRA5ktAYYS69VH8bRb_jlciy6ME5aZorNuduIYMwAFhyVOGs2Ou1tIx6548kujDN-evgnYCACMhi-qn4c1LknLTO_Rmff_JzxcIXCWcqL5ePcPQ1teKKWTU8citiWh9FGcp5Ff1ym12FVIUmNHD7vsyd9thIuWJnUjLZduvM7OKo7_EaVGcvfz5ZXlN7GNAo7JgFjExpr-x7K08JCoYLUrUnPj8zeT4C'

export default function App() {
  const [messages, setMessages] = useState([{ id: 0, kind: 'assistant-text', text: GREETING }])
  const [phase, setPhase] = useState('compose') // compose | busy | review | done
  const [busyLabel, setBusyLabel] = useState('')
  const [chips, setChips] = useState([])
  const [pendingQuery, setPendingQuery] = useState('')
  const idRef = useRef(1)
  const bottomRef = useRef(null)

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, phase])

  const push = (msg) => setMessages((m) => [...m, { id: idRef.current++, ...msg }])
  const reset = () => { setMessages([{ id: idRef.current++, kind: 'assistant-text', text: GREETING }]); setChips([]); setPendingQuery(''); setPhase('compose') }

  async function doRecommend(query, ingredients) {
    setBusyLabel('Finding the best recipe...'); setPhase('busy')
    try {
      const res = await recommend(query, ingredients)
      push(
        res.status === 'no_safe_candidate'
          ? { kind: 'nosafe', res }
          : res.status === 'out_of_scope'
            ? { kind: 'out-of-scope', res }
            : { kind: 'result', res, userIngredients: ingredients },
      )
    } catch (e) {
      push({ kind: 'error', text: `Server error: ${e.message}. Check the backend, Qdrant and Ollama.` })
    } finally { setPhase('done') }
  }

  async function handleSend(text, files, previews) {
    push({ kind: 'user', text, imageUrls: previews })
    setPendingQuery(text)
    if (files && files.length) {
      setBusyLabel(files.length > 1 ? `Detecting ingredients from ${files.length} images...` : 'Detecting ingredients...')
      setPhase('busy')
      try {
        const batches = []
        for (const file of files) batches.push(...await detectIngredients(file))
        const merged = new Map()
        for (const d of batches) {
          const name = d.canonical
          if (!name) continue
          const prev = merged.get(name)
          if (!prev || (d.confidence ?? 0) > (prev.confidence ?? 0)) {
            merged.set(name, { name, confidence: d.confidence, kept: true })
          }
        }
        setChips([...merged.values()].sort((a, b) => a.name.localeCompare(b.name)))
        setPhase('review')
      } catch (e) { push({ kind: 'error', text: `Could not read the image: ${e.message}` }); setPhase('compose') }
    } else {
      await doRecommend(text, [])
    }
  }

  function recommendFromReview() {
    const kept = chips.filter((c) => c.kept).map((c) => c.name)
    push({ kind: 'user', text: kept.length ? `Ingredients: ${kept.join(', ')}` : 'No changes' })
    doRecommend(pendingQuery, kept)
  }
  const toggleChip = (i) => setChips((cs) => cs.map((c, j) => (j === i ? { ...c, kept: !c.kept } : c)))
  const removeChip = (i) => setChips((cs) => cs.filter((_, j) => j !== i))
  const addChip = (name) => setChips((cs) => [...cs, { name, confidence: null, kept: true }])

  return (
    <div className="min-h-screen flex flex-col items-center pb-32 text-on-surface">
      {/* Background */}
      <div className="fixed inset-0 -z-10 overflow-hidden">
        <img alt="" className="w-full h-full object-cover blur-sm scale-105" src={BG} />
        <div className="absolute inset-0 bg-background/60 backdrop-blur-[2px]"></div>
      </div>

      {/* Header */}
      <header className="fixed top-0 left-0 w-full z-[60] backdrop-blur-md bg-surface-container-lowest/80 border-b border-surface-container-high py-3 flex justify-center items-center" style={{ borderBottomLeftRadius: '24px', borderBottomRightRadius: '24px' }}>
        <div className="w-full max-w-[680px] px-md flex justify-between items-center">
          <div className="flex items-center gap-2 text-primary font-headline-md text-[24px]"><span className="material-symbols-outlined text-[28px]">restaurant_menu</span><span className="font-bold">Recipe Assistant</span></div>
          <button onClick={reset} className="flex items-center gap-1 px-4 py-1.5 rounded-full border border-outline-variant text-primary hover:bg-primary/10 transition-colors font-label-md"><span className="material-symbols-outlined text-[20px]">refresh</span><span>Start Over</span></button>
        </div>
      </header>

      {/* Thread */}
      <main className="w-full max-w-[680px] px-md flex flex-col gap-lg" style={{ paddingTop: '120px', paddingBottom: '40px' }}>
        {messages.map((m) => <Message key={m.id} m={m} />)}
        {phase === 'busy' ? <Loading label={busyLabel} /> : null}
        {phase === 'review' ? (
          <ReviewChips chips={chips} onToggle={toggleChip} onRemove={removeChip} onAdd={addChip} onRecommend={recommendFromReview} />
        ) : null}
        {phase === 'done' ? (
          <div className="flex justify-center pt-2">
            <button onClick={reset} className="flex items-center gap-2 px-6 py-3 bg-surface-container-lowest border border-surface-container-high hover:bg-surface-container rounded-full font-label-md text-on-surface-variant transition-colors shadow-low"><span className="material-symbols-outlined">refresh</span> New recipe</button>
          </div>
        ) : null}
        <div ref={bottomRef} />
      </main>

      <Composer onSend={handleSend} busy={phase === 'busy'} />
    </div>
  )
}

function Message({ m }) {
  if (m.kind === 'assistant-text') return <AssistantText>{m.text}</AssistantText>
  if (m.kind === 'user') return <UserTurn text={m.text} imageUrls={m.imageUrls || (m.imageUrl ? [m.imageUrl] : [])} />
  if (m.kind === 'result') return <RecipeCard res={m.res} userIngredients={m.userIngredients} />
  if (m.kind === 'nosafe') return <NoSafeCard res={m.res} />
  if (m.kind === 'out-of-scope') return <OutOfScopeCard res={m.res} />
  if (m.kind === 'error') return <AssistantText>{m.text}</AssistantText>
  return null
}
