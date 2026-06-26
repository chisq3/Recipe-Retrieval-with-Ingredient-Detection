import { useState } from 'react'

const na = (v) => (v === null || v === undefined || v === '') ? 'N/A' : v

export const Icon = ({ name, className = '', style }) => (
  <span className={`material-symbols-outlined ${className}`} style={style}>{name}</span>
)

const Avatar = ({ faded }) => (
  <div className={`w-8 h-8 rounded-full bg-primary-container flex items-center justify-center shrink-0${faded ? ' opacity-0' : ''}`}>
    <span className="material-symbols-outlined text-on-primary-container text-[20px]">restaurant_menu</span>
  </div>
)

export function AssistantText({ children }) {
  return (
    <div className="flex items-start gap-md w-full">
      <Avatar />
      <div className="flex flex-col gap-sm max-w-[85%]"><div className="bg-surface-container-lowest/80 backdrop-blur-md p-md rounded-xl rounded-tl-none shadow-low text-on-surface font-body-md">{children}</div></div>
    </div>
  )
}

export function UserTurn({ text, imageUrl }) {
  return (
    <div className="flex flex-col items-end gap-sm w-full">
      <div className="flex items-start gap-md max-w-[85%] justify-end">
        <div className="flex flex-col items-end gap-sm">
          {text ? <div className="bg-surface-container-high rounded-xl rounded-tr-none p-md shadow-low text-on-surface font-body-md">{text}</div> : null}
          {imageUrl ? <div className="h-20 w-20 rounded-lg overflow-hidden border border-outline-variant shadow-low shrink-0"><img className="w-full h-full object-cover" src={imageUrl} alt="" /></div> : null}
        </div>
        <div className="w-8 h-8 rounded-full bg-surface-container flex items-center justify-center shrink-0"><span className="material-symbols-outlined text-on-surface-variant text-[20px]">person</span></div>
      </div>
    </div>
  )
}

export function Loading({ label }) {
  return (
    <div className="flex items-start gap-md w-full">
      <Avatar />
      <div className="flex items-center gap-2 text-on-surface-variant font-body-md"><span className="material-symbols-outlined animate-spin text-primary">progress_activity</span>{label}</div>
    </div>
  )
}

export function ReviewChips({ chips, onToggle, onRemove, onAdd, onRecommend }) {
  const [text, setText] = useState('')
  const add = () => { const v = text.trim(); if (v) { onAdd(v); setText('') } }
  return (
    <div className="flex items-start gap-md w-full">
      <Avatar />
      <div className="flex flex-col gap-sm max-w-[85%] w-full">
        <div className="text-on-surface font-body-md">
          {chips.length ? 'I detected these ingredients — keep the ones you have' : "I couldn't detect any ingredients — add them manually, then hit Recommend"}
        </div>
        <div className="flex flex-wrap gap-sm">
          {chips.map((c, i) => (
            <button key={i} onClick={() => onToggle(i)}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 bg-primary-container/10 border border-primary/20 rounded-full hover:bg-primary-container/20 transition-colors${c.kept ? '' : ' opacity-40 line-through'}`}>
              <span className="font-label-md text-primary">{c.name}</span>
              {c.confidence != null ? <span className="font-label-sm text-primary/70">· {Math.round(c.confidence * 100)}%</span> : null}
              <span onClick={(e) => { e.stopPropagation(); onRemove(i) }} className="material-symbols-outlined text-primary text-[14px]">close</span>
            </button>
          ))}
        </div>
        <div className="flex gap-2 max-w-xs mt-1">
          <input value={text} onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && add()}
            placeholder="Add ingredient..." className="flex-1 bg-surface-container-lowest border border-outline-variant rounded-lg py-2 px-3 font-body-md focus:border-primary focus:outline-none" />
          <button onClick={add} className="px-4 rounded-lg bg-surface-container border border-outline-variant font-label-md text-on-surface">Add</button>
        </div>
        <button onClick={onRecommend} className="mt-sm inline-flex items-center justify-center gap-2 px-5 py-2.5 bg-primary text-on-primary rounded-full font-label-md hover:bg-primary/90 transition-colors self-start">
          <span className="material-symbols-outlined text-[20px]">auto_awesome</span>Recommend
        </button>
      </div>
    </div>
  )
}

function NutritionToggle({ n }) {
  const [open, setOpen] = useState(false)
  const cell = (label, val, unit) => (
    <div className="p-2"><div className="text-label-sm text-outline">{label}</div><div className="font-label-md">{val == null ? 'N/A' : `${val}${unit || ''}`}</div></div>
  )
  return (
    <div className="border border-surface-container-high rounded-lg overflow-hidden bg-surface-container-low/50 backdrop-blur-sm" style={{ borderRadius: '12px' }}>
      <div onClick={() => setOpen(!open)} className="p-md cursor-pointer hover:bg-surface-container transition-colors flex items-center justify-between">
        <div className="flex items-center gap-2"><span className="material-symbols-outlined text-primary text-[24px]">nutrition</span><div className="font-label-md text-on-surface text-[16px]">Nutrition Facts</div></div>
        <span className={`material-symbols-outlined text-on-surface-variant transition-transform${open ? ' rotate-180' : ''}`}>expand_more</span>
      </div>
      {open ? (
        <div className="p-md border-t border-surface-container-high bg-surface-container-lowest/50"><div className="grid grid-cols-3 gap-2 text-center">
          {cell('Calories', n.calories, '')}{cell('Protein', n.protein_g, 'g')}{cell('Total Fat', n.fat_g, 'g')}
          {cell('Saturated Fat', n.saturated_fat_g, 'g')}{cell('Cholesterol', n.cholesterol_mg, 'mg')}{cell('Sodium', n.sodium_mg, 'mg')}
          {cell('Total Carbs', n.carbohydrate_g, 'g')}{cell('Dietary Fiber', n.fiber_g, 'g')}{cell('Sugars', n.sugar_g, 'g')}
        </div></div>
      ) : null}
    </div>
  )
}

function QuickList({ items, have }) {
  if (!items || !items.length) return <li className="text-on-surface-variant font-body-md">—</li>
  return items.map((x, i) => have ? (
    <li key={i} className="flex items-center gap-3"><div className="w-5 h-5 rounded border border-success bg-success flex items-center justify-center shrink-0"><span className="material-symbols-outlined text-on-primary text-[14px]">check</span></div><span className="font-body-md text-on-surface">{x}</span></li>
  ) : (
    <li key={i} className="flex items-center gap-3"><div className="w-5 h-5 rounded border border-outline-variant shrink-0"></div><span className="font-body-md text-on-surface">{x}</span></li>
  ))
}

export function RecipeCard({ res, userIngredients }) {
  const [showOriginal, setShowOriginal] = useState(false)
  const f = res.recipe_facts || {}, t = f.time_minutes || {}, n = f.nutrition || {}, r = f.rating || {}, ev = res.evidence || {}
  const feasible = res.feasibility && !String(res.feasibility).includes('not_recommended')
  // Vietnamese ↔ original toggle (title always stays original)
  const hasVi = !!(res.ingredients_full_vi || res.instructions_vi || (res.adapted_steps && res.adapted_steps.length))
  const ingredients = showOriginal ? (res.ingredients_full || []) : (res.ingredients_full_vi || res.ingredients_full || [])
  const stepsVi = (res.adapted_steps && res.adapted_steps.length) ? res.adapted_steps : (res.instructions_vi || res.instructions || [])
  const stepsEn = (res.instructions && res.instructions.length) ? res.instructions : (res.adapted_steps || [])
  const steps = showOriginal ? stepsEn : stepsVi
  const shopping = (res.shopping_list && res.shopping_list.length ? res.shopping_list : (res.missing_core_ingredients || []))
  const cuisines = res.cuisine_tags ? String(res.cuisine_tags).split('|').map((s) => s.trim()).filter(Boolean) : []
  const u = ev.understood_request || {}
  const safe = [], soft = []
  if (u.diet) safe.push('diet: ' + u.diet)
  ;(u.method_exclude || []).forEach((m) => safe.push('no ' + m))
  ;(u.exclude || []).forEach((m) => safe.push('exclude: ' + m))
  if (u.cuisine) soft.push('cuisine: ' + u.cuisine)
  if (u.meal_type) soft.push('meal: ' + u.meal_type)
  if (u.max_time) soft.push('≤ ' + u.max_time + 'm')
  const stat = (icon, label, val) => (
    <div className="flex flex-col items-center text-center gap-1"><span className="material-symbols-outlined text-outline text-[20px]">{icon}</span><span className="font-label-sm text-outline">{label}</span><span className="font-label-md text-on-surface">{val}</span></div>
  )
  return (
    <div className="flex items-start gap-md w-full">
      <Avatar faded />
      <div className="w-full max-w-[85%]">
        <div className="bg-surface-container-lowest rounded-[12px] shadow-low overflow-hidden border border-surface-container-high w-full bg-surface-container-lowest/80 backdrop-blur-md">
          <div className="w-full h-48 relative bg-surface-container">
            {res.image_url ? <img className="w-full h-full object-cover" src={res.image_url} alt={res.recipe_title} /> : <div className="w-full h-full flex items-center justify-center text-outline"><span className="material-symbols-outlined text-[48px]">restaurant</span></div>}
            <div className="absolute top-md right-md flex gap-sm">
              <button className="w-8 h-8 rounded-full bg-surface-container-lowest/80 backdrop-blur flex items-center justify-center shadow-low text-on-surface hover:text-primary transition-colors"><span className="material-symbols-outlined text-[20px]">bookmark_border</span></button>
              <button className="w-8 h-8 rounded-full bg-surface-container-lowest/80 backdrop-blur flex items-center justify-center shadow-low text-on-surface hover:text-primary transition-colors"><span className="material-symbols-outlined text-[20px]">share</span></button>
            </div>
          </div>
          <div className="p-lg flex flex-col gap-lg">
            <div className="flex flex-col gap-sm">
              <div className="flex items-center gap-sm flex-wrap">
                {cuisines.map((c) => <span key={c} className="px-2 py-1 rounded bg-surface-container font-label-sm text-on-surface-variant uppercase tracking-wider">{c}</span>)}
                <span className={`px-2 py-1 rounded font-label-sm inline-flex items-center gap-1 ${feasible ? 'bg-success/10 text-success' : 'bg-secondary-container/30 text-secondary'}`}><span className="material-symbols-outlined text-[14px]">{feasible ? 'check_circle' : 'info'}</span> {feasible ? 'Feasible' : 'Needs review'}</span>
                {r.average != null ? (
                  <div className="ml-auto flex items-center gap-1 text-secondary-container"><span className="material-symbols-outlined text-[16px]" style={{ fontVariationSettings: "'FILL' 1" }}>star</span><span className="font-label-md text-on-surface">{r.average}</span>{r.review_count != null ? <span className="font-label-sm text-on-surface-variant">({r.review_count})</span> : null}</div>
                ) : null}
              </div>
              <h2 className="font-headline-md text-on-surface text-[28px]">{res.recipe_title || 'Công thức'}</h2>
            </div>

            {hasVi ? (
              <button onClick={() => setShowOriginal(!showOriginal)} className="self-start inline-flex items-center gap-1 text-primary font-label-sm hover:underline">
                <span className="material-symbols-outlined text-[16px]">translate</span>
                {showOriginal ? 'Show Vietnamese' : 'Show original (EN)'}
              </button>
            ) : null}

            {res.why_recommended ? <div className="bg-primary/5 border-l-4 border-primary p-md rounded-r-lg"><p className="font-body-md text-on-surface-variant italic">{res.why_recommended}</p></div> : null}
            {res.warning ? <div className="flex items-start gap-2 bg-secondary-container/25 border border-secondary-fixed-dim rounded-lg p-md"><span className="material-symbols-outlined text-secondary text-[20px]">warning</span><p className="font-body-md text-on-surface">{res.warning}</p></div> : null}

            <div className="grid grid-cols-4 gap-md py-md border-y border-surface-container-high">
              {stat('skillet', 'Prep', t.prep == null ? 'N/A' : t.prep + 'm')}
              {stat('local_fire_department', 'Cook', t.cook == null ? 'N/A' : t.cook + 'm')}
              {stat('schedule', 'Total', t.total == null ? 'N/A' : t.total + 'm')}
              {stat('restaurant', 'Serves', na(f.servings))}
            </div>

            <NutritionToggle n={n} />

            <div className="flex flex-col gap-md"><h3 className="font-title-lg text-primary text-[22px]">Quick View</h3><div className="grid grid-cols-2 gap-lg">
              <div className="flex flex-col gap-sm"><div className="font-label-sm text-success uppercase tracking-wider">You have</div><ul className="flex flex-col gap-2"><QuickList items={userIngredients} have /></ul></div>
              <div className="flex flex-col gap-sm"><div className="font-label-sm text-error uppercase tracking-wider">Need to buy</div><ul className="flex flex-col gap-2"><QuickList items={shopping} /></ul></div>
            </div></div>

            <div className="flex flex-col gap-md"><h3 className="font-title-lg text-primary text-[22px]">Ingredients</h3>
              <ul className="grid grid-cols-2 gap-x-lg gap-y-2 list-disc pl-5 font-body-md text-on-surface">
                {ingredients.length ? ingredients.map((x, i) => <li key={i}>{x}</li>) : <li className="text-on-surface-variant">—</li>}
              </ul>
            </div>

            <div className="flex flex-col gap-md"><h3 className="font-title-lg text-primary text-[22px]">Instructions</h3><div className="flex flex-col gap-md">
              {steps.length ? steps.map((s, i) => (
                <div key={i} className="flex gap-md"><div className="w-6 h-6 rounded-full bg-primary text-on-primary font-label-sm flex items-center justify-center shrink-0 mt-0.5">{i + 1}</div><div className="font-body-md text-on-surface">{s}</div></div>
              )) : <div className="font-body-md text-on-surface-variant">(no instructions)</div>}
            </div></div>

            <details className="border border-surface-container-high bg-surface-container-low/60 backdrop-blur-sm group" style={{ borderRadius: '12px' }}>
              <summary className="p-md cursor-pointer flex items-center justify-between list-none"><div className="flex items-center gap-2"><span className="material-symbols-outlined text-primary text-[22px]">analytics</span><span className="font-label-md text-on-surface text-[16px]">Evidence &amp; Decision</span></div><span className="material-symbols-outlined text-on-surface-variant transition-transform group-open:rotate-180">expand_more</span></summary>
              <div className="p-md border-t border-surface-container-high flex flex-col gap-md">
                <div>
                  <div className="font-label-sm text-outline uppercase tracking-wider mb-2">Understood request</div>
                  <div className="flex flex-wrap gap-2">
                    {(!safe.length && !soft.length) ? <span className="font-label-sm text-on-surface-variant">(no special constraints)</span> : null}
                    {safe.map((c) => <span key={c} className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-success/10 text-success font-label-sm"><span className="material-symbols-outlined text-[14px]">verified_user</span>{c}</span>)}
                    {soft.map((c) => <span key={c} className="px-2 py-1 rounded-full bg-surface-container text-on-surface-variant font-label-sm">{c}</span>)}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-x-lg gap-y-2 font-label-sm text-on-surface-variant">
                  <div className="flex justify-between border-b border-surface-container-high/60 pb-1"><span>Source</span><span className="font-bold text-on-surface">{na(res.candidate_source)}</span></div>
                  <div className="flex justify-between border-b border-surface-container-high/60 pb-1"><span>Retrieval rank</span><span className="font-bold text-on-surface">{na(res.selected_rank)}</span></div>
                  <div className="flex justify-between border-b border-surface-container-high/60 pb-1"><span>Request constraints</span><span className="font-bold text-on-surface">{res.extraction_source === 'human_confirmed' ? 'Human confirmed' : 'Model extracted'}</span></div>
                  <div className="flex justify-between border-b border-surface-container-high/60 pb-1"><span>Rejected by gate</span><span className="font-bold text-error">{na(ev.rejected_by_gate)}</span></div>
                  <div className="flex justify-between border-b border-surface-container-high/60 pb-1"><span>Issues repaired</span><span className="font-bold text-on-surface">{na(ev.validation_issues_repaired)}</span></div>
                  <div className="flex justify-between"><span>Processing time</span><span className="font-bold text-on-surface">{(ev.timings_seconds && ev.timings_seconds.total != null) ? ev.timings_seconds.total + 's' : 'N/A'}</span></div>
                </div>
              </div>
            </details>
          </div>
        </div>
      </div>
    </div>
  )
}

export function NoSafeCard({ res }) {
  return (
    <div className="flex items-start gap-md w-full">
      <Avatar />
      <div className="w-full max-w-[85%]">
        <div className="bg-error-container/40 border border-error/30 rounded-[12px] p-lg flex flex-col gap-md">
          <div className="flex items-start gap-3"><span className="material-symbols-outlined text-error text-[28px]">do_not_disturb_on</span><div><h3 className="font-headline-md text-on-surface text-[22px]">No recipe satisfies the constraints</h3><p className="font-body-md text-on-surface-variant">This isn't an error — the safety gate checked every candidate and none satisfied your constraints.</p></div></div>
          <div className="grid grid-cols-2 gap-3"><div className="bg-surface-container-lowest rounded-lg p-md text-center border border-surface-container-high"><div className="font-headline-md text-[24px]">{na(res.considered_count)}</div><div className="font-label-sm text-outline">considered</div></div><div className="bg-error-container/40 rounded-lg p-md text-center border border-error/20"><div className="font-headline-md text-[24px] text-error">{na(res.rejected_count)}</div><div className="font-label-sm text-outline">rejected by gate</div></div></div>
          <div><div className="font-label-sm text-outline uppercase tracking-wider mb-1">Violated constraints</div><div className="flex flex-wrap gap-2">
            {(res.violated_constraint_types && res.violated_constraint_types.length) ? res.violated_constraint_types.map((c) => <span key={c} className="inline-flex items-center gap-1 px-3 py-1 rounded-full bg-surface-container-lowest border border-error/30 text-error font-label-sm"><span className="material-symbols-outlined text-[14px]">block</span>{c}</span>) : <span className="font-label-sm text-on-surface-variant">—</span>}
          </div></div>
        </div>
      </div>
    </div>
  )
}

export function OutOfScopeCard({ res }) {
  const total = res.evidence?.timings_seconds?.total
  return (
    <div className="flex items-start gap-md w-full">
      <Avatar />
      <div className="w-full max-w-[85%]">
        <div className="bg-surface-container-lowest/80 border border-surface-container-high rounded-[12px] p-lg flex flex-col gap-md shadow-low">
          <div className="flex items-start gap-3">
            <span className="material-symbols-outlined text-primary text-[28px]">restaurant_menu</span>
            <div>
              <h3 className="font-headline-md text-on-surface text-[22px]">Ask me about recipes</h3>
              <p className="font-body-md text-on-surface-variant">
                {res.message || 'I can help with ingredients, cooking constraints, and meal ideas.'}
              </p>
            </div>
          </div>
          <div className="font-label-sm text-on-surface-variant">
            Try something like <span className="font-bold text-on-surface">tomato pasta dinner</span> or attach a photo of ingredients.
          </div>
          {total != null ? (
            <div className="font-label-sm text-outline">Checked request in {total}s</div>
          ) : null}
        </div>
      </div>
    </div>
  )
}

export function Composer({ onSend, busy }) {
  const [text, setText] = useState('')
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const send = () => {
    if (busy) return
    if (!text.trim() && !file) return
    onSend(text.trim(), file, preview)
    setText(''); setFile(null); setPreview(null)
  }
  const pick = (e) => { const f = e.target.files?.[0]; if (f) { setFile(f); setPreview(URL.createObjectURL(f)) } }
  return (
    <div className="fixed bottom-0 left-0 w-full z-50 flex flex-col items-center px-md pb-lg pt-md bg-gradient-to-t from-background via-background to-transparent pointer-events-none">
      {preview ? (
        <div className="w-full max-w-[680px] mb-2 flex items-center gap-2 pointer-events-auto">
          <img src={preview} className="h-12 w-12 rounded-lg object-cover border border-outline-variant" alt="" />
          <button onClick={() => { setFile(null); setPreview(null) }} className="font-label-sm text-error">Remove</button>
        </div>
      ) : null}
      <div className="w-full max-w-[680px] bg-surface-container-lowest rounded-full shadow-high border border-surface-container-high flex items-center p-2 pointer-events-auto transition-shadow focus-within:shadow-md focus-within:border-primary/50">
        <label className="w-10 h-10 flex items-center justify-center text-on-surface-variant hover:bg-surface-container rounded-full transition-colors shrink-0 cursor-pointer" title="Attach an ingredient photo">
          <span className="material-symbols-outlined text-[24px]">attach_file</span>
          <input type="file" accept="image/*" className="hidden" onChange={pick} />
        </label>
        <input value={text} onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && send()} disabled={busy}
          className="flex-1 bg-transparent border-none focus:ring-0 focus:outline-none text-body-md text-on-surface placeholder:text-outline px-sm py-2" placeholder="What do you want to cook? (e.g. vegetarian pho, no oven)" type="text" />
        <button onClick={send} disabled={busy} className="w-10 h-10 flex items-center justify-center bg-primary text-on-primary hover:bg-primary/90 rounded-full transition-colors shrink-0 shadow-sm disabled:opacity-60">
          <span className="material-symbols-outlined text-[20px]">arrow_upward</span>
        </button>
      </div>
    </div>
  )
}
