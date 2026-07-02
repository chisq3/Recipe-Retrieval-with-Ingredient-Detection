import { useState } from 'react'

const na = (v) => (v === null || v === undefined || v === '') ? 'N/A' : v

export const Icon = ({ name, className = '', style }) => (
  <span className={`material-symbols-outlined ${className}`} style={style}>{name}</span>
)

function feasibilityBadge(feasibility) {
  if (feasibility === 'cookable_as_is') {
    return {
      label: 'Feasible',
      icon: 'check_circle',
      className: 'bg-success/10 text-success',
    }
  }
  if (feasibility === 'cookable_with_minor_adjustment') {
    return {
      label: 'Needs minor addition',
      icon: 'add_shopping_cart',
      className: 'bg-secondary-container/30 text-secondary',
    }
  }
  return {
    label: 'Needs review',
    icon: 'info',
    className: 'bg-secondary-container/30 text-secondary',
  }
}

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

export function UserTurn({ text, imageUrls = [] }) {
  return (
    <div className="flex flex-col items-end gap-sm w-full">
      <div className="flex items-start gap-md max-w-[85%] justify-end">
        <div className="flex flex-col items-end gap-sm">
          {text ? <div className="bg-surface-container-high rounded-xl rounded-tr-none p-md shadow-low text-on-surface font-body-md">{text}</div> : null}
          {imageUrls.length ? (
            <div className="flex flex-wrap justify-end gap-2 max-w-[260px]">
              {imageUrls.map((imageUrl, i) => (
                <div key={i} className="h-20 w-20 rounded-lg overflow-hidden border border-outline-variant shadow-low shrink-0">
                  <img className="w-full h-full object-cover" src={imageUrl} alt="" />
                </div>
              ))}
            </div>
          ) : null}
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
          {chips.length ? 'I detected these ingredients. Keep the ones you have.' : "I couldn't detect any ingredients. Add them manually, then hit Recommend."}
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

function RecipeCardOld({ res, userIngredients }) {
  const [showOriginal, setShowOriginal] = useState(false)
  const [open, setOpen] = useState(false)
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
  const compactList = (items, empty = 'None') => (
    items && items.length ? `${items.slice(0, 5).join(', ')}${items.length > 5 ? `, +${items.length - 5} more` : ''}` : empty
  )
  return (
    <div className="flex items-start gap-md w-full">
      <Avatar faded />
      <div className="w-full max-w-[85%]">
        <div className="bg-surface-container-lowest rounded-[12px] shadow-low overflow-hidden border border-surface-container-high w-full bg-surface-container-lowest/80 backdrop-blur-md">
          <div className="w-full h-48 relative bg-surface-container">
            {res.image_url ? <img className="w-full h-full object-cover" src={res.image_url} alt={res.recipe_title} /> : <div className="w-full h-full flex items-center justify-center text-outline"><span className="material-symbols-outlined text-[48px]">restaurant</span></div>}
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
              <h2 className="font-headline-md text-on-surface text-[28px]">{res.recipe_title || 'Recipe'}</h2>
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

            <div className="grid grid-cols-2 gap-md">
              <div className="rounded-lg bg-success/5 border border-success/20 p-md">
                <div className="font-label-sm text-success uppercase tracking-wider mb-1">Available</div>
                <div className="font-body-md text-on-surface">{compactList(userIngredients)}</div>
              </div>
              <div className="rounded-lg bg-surface-container-low border border-surface-container-high p-md">
                <div className="font-label-sm text-error uppercase tracking-wider mb-1">Need to buy</div>
                <div className="font-body-md text-on-surface">{compactList(shopping)}</div>
              </div>
            </div>

            <button onClick={() => setOpen(!open)} className="inline-flex items-center justify-center gap-2 rounded-full bg-primary text-on-primary px-5 py-2.5 font-label-md hover:bg-primary/90 transition-colors self-start">
              <span className="material-symbols-outlined text-[20px]">{open ? 'expand_less' : 'read_more'}</span>
              {open ? 'Hide full recipe' : 'View full recipe'}
            </button>

            <div className={open ? 'flex flex-col gap-lg border-t border-surface-container-high pt-lg' : 'hidden'}>
              <NutritionToggle n={n} />

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

            </div>

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

export function RecipeCard({ res, userIngredients }) {
  const [open, setOpen] = useState(false)
  const [showOriginal, setShowOriginal] = useState(false)
  const f = res.recipe_facts || {}, t = f.time_minutes || {}, n = f.nutrition || {}, r = f.rating || {}, ev = res.evidence || {}
  const badge = feasibilityBadge(res.feasibility)
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
  if (u.max_time) soft.push('<= ' + u.max_time + 'm')
  const fullList = (items, empty = 'None') => (items && items.length ? items.join(', ') : empty)

  const stat = (icon, label, val) => (
    <div className="rounded-lg bg-surface-container-low border border-surface-container-high p-2 text-center">
      <span className="material-symbols-outlined text-outline text-[18px]">{icon}</span>
      <div className="font-label-sm text-outline">{label}</div>
      <div className="font-label-md text-on-surface">{val}</div>
    </div>
  )

  const image = res.image_url ? (
    <img className="w-full h-full object-cover" src={res.image_url} alt={res.recipe_title} />
  ) : (
    <div className="w-full h-full flex items-center justify-center text-outline bg-surface-container">
      <span className="material-symbols-outlined text-[56px]">restaurant</span>
    </div>
  )

  return (
    <div className="flex items-start gap-md w-full">
      <Avatar faded />
      <div className="w-full max-w-[85%]">
        <button onClick={() => setOpen(true)} className="w-full text-left bg-surface-container-lowest rounded-[12px] shadow-low overflow-hidden border border-surface-container-high bg-surface-container-lowest/85 backdrop-blur-md hover:shadow-high hover:border-primary/30 transition-all">
          <div className="w-full h-56 relative bg-surface-container">{image}</div>
          <div className="p-lg flex flex-col gap-md">
            <div className="flex items-center gap-sm flex-wrap">
              <span className={`px-2 py-1 rounded font-label-sm inline-flex items-center gap-1 ${badge.className}`}>
                <span className="material-symbols-outlined text-[14px]">{badge.icon}</span>
                {badge.label}
              </span>
              {cuisines.slice(0, 2).map((c) => <span key={c} className="px-2 py-1 rounded bg-surface-container font-label-sm text-on-surface-variant uppercase tracking-wider">{c}</span>)}
              {r.average != null ? (
                <div className="ml-auto flex items-center gap-1 text-secondary-container">
                  <span className="material-symbols-outlined text-[16px]" style={{ fontVariationSettings: "'FILL' 1" }}>star</span>
                  <span className="font-label-md text-on-surface">{r.average}</span>
                  {r.review_count != null ? <span className="font-label-sm text-on-surface-variant">({r.review_count})</span> : null}
                </div>
              ) : null}
            </div>
            <h2 className="font-headline-md text-on-surface text-[28px] leading-tight">{res.recipe_title || 'Recipe'}</h2>
          </div>
        </button>
      </div>

      {open ? (
        <div className="fixed inset-0 z-[100] bg-black/45 backdrop-blur-sm flex items-center justify-center p-3 sm:p-6" onClick={() => setOpen(false)}>
          <div className="w-full max-w-[1320px] max-h-[94vh] bg-surface-container-lowest rounded-[18px] shadow-high overflow-hidden border border-surface-container-high" onClick={(e) => e.stopPropagation()}>
            <div className="max-h-[94vh] overflow-y-auto">
            <div className="sticky top-0 z-10 flex items-center justify-between gap-md px-lg py-md border-b border-surface-container-high bg-surface-container-lowest">
              <div className="min-w-0">
                <div className="font-label-sm text-outline uppercase tracking-wider">Selected recipe</div>
                <h2 className="font-headline-md text-on-surface text-[24px] leading-tight truncate">{res.recipe_title || 'Recipe'}</h2>
              </div>
              <button onClick={() => setOpen(false)} className="w-10 h-10 rounded-full hover:bg-surface-container flex items-center justify-center text-on-surface-variant">
                <span className="material-symbols-outlined">close</span>
              </button>
            </div>

            <div className="grid lg:grid-cols-[390px_minmax(0,1fr)]">
              <aside className="bg-surface-container-low border-r border-surface-container-high">
                <div className="h-56 bg-surface-container">{image}</div>
                <div className="p-lg flex flex-col gap-md">
                  <div className="flex flex-wrap gap-sm">
                    <span className={`px-2 py-1 rounded font-label-sm inline-flex items-center gap-1 ${badge.className}`}>
                      <span className="material-symbols-outlined text-[14px]">{badge.icon}</span>
                      {badge.label}
                    </span>
                    {cuisines.map((c) => <span key={c} className="px-2 py-1 rounded bg-surface-container font-label-sm text-on-surface-variant uppercase tracking-wider">{c}</span>)}
                  </div>

                  <div className="grid grid-cols-2 gap-sm">
                    {stat('skillet', 'Prep', t.prep == null ? 'N/A' : t.prep + 'm')}
                    {stat('local_fire_department', 'Cook', t.cook == null ? 'N/A' : t.cook + 'm')}
                    {stat('schedule', 'Total', t.total == null ? 'N/A' : t.total + 'm')}
                    {stat('restaurant', 'Serves', na(f.servings))}
                  </div>

                  <NutritionToggle n={n} />
                </div>
              </aside>

              <main className="p-lg flex flex-col gap-lg">
                {hasVi ? (
                  <button onClick={() => setShowOriginal(!showOriginal)} className="self-start inline-flex items-center gap-1 text-primary font-label-sm hover:underline">
                    <span className="material-symbols-outlined text-[16px]">translate</span>
                    {showOriginal ? 'Show Vietnamese' : 'Show original (EN)'}
                  </button>
                ) : null}

                <section className="flex flex-col gap-md">
                  <h3 className="font-title-lg text-primary text-[22px]">Quick View</h3>
                  <div className="grid md:grid-cols-2 gap-md">
                    <div className="rounded-lg bg-success/5 border border-success/20 p-md">
                      <div className="font-label-sm text-success uppercase tracking-wider mb-1">Available</div>
                      <div className="font-body-md text-on-surface leading-relaxed">{fullList(userIngredients)}</div>
                    </div>
                    <div className="rounded-lg bg-surface-container-low border border-surface-container-high p-md">
                      <div className="font-label-sm text-error uppercase tracking-wider mb-1">Need to buy</div>
                      <div className="font-body-md text-on-surface leading-relaxed">{fullList(shopping)}</div>
                    </div>
                  </div>
                </section>

                {res.why_recommended ? (
                  <section className="bg-primary/5 border-l-4 border-primary p-md rounded-r-lg">
                    <div className="font-label-sm text-primary uppercase tracking-wider mb-1">Recommendation reason</div>
                    <p className="font-body-md text-on-surface-variant italic">{res.why_recommended}</p>
                  </section>
                ) : null}

                {res.warning ? (
                  <section className="flex items-start gap-2 bg-secondary-container/25 border border-secondary-fixed-dim rounded-lg p-md">
                    <span className="material-symbols-outlined text-secondary text-[20px]">warning</span>
                    <div>
                      <div className="font-label-sm text-secondary uppercase tracking-wider mb-1">Warning</div>
                      <p className="font-body-md text-on-surface">{res.warning}</p>
                    </div>
                  </section>
                ) : null}

                <div className="grid xl:grid-cols-[0.9fr_1.1fr] gap-lg">
                  <section className="flex flex-col gap-md">
                    <h3 className="font-title-lg text-primary text-[22px]">Ingredients</h3>
                    <ul className="grid sm:grid-cols-2 xl:grid-cols-1 gap-x-lg gap-y-2 list-disc pl-5 font-body-md text-on-surface">
                      {ingredients.length ? ingredients.map((x, i) => <li key={i}>{x}</li>) : <li className="text-on-surface-variant">-</li>}
                    </ul>
                  </section>

                  <section className="flex flex-col gap-md">
                    <h3 className="font-title-lg text-primary text-[22px]">Instructions</h3>
                    <div className="flex flex-col gap-md">
                      {steps.length ? steps.map((s, i) => (
                        <div key={i} className="flex gap-md">
                          <div className="w-6 h-6 rounded-full bg-primary text-on-primary font-label-sm flex items-center justify-center shrink-0 mt-0.5">{i + 1}</div>
                          <div className="font-body-md text-on-surface">{s}</div>
                        </div>
                      )) : <div className="font-body-md text-on-surface-variant">(no instructions)</div>}
                    </div>
                  </section>
                </div>

                <details className="border border-surface-container-high bg-surface-container-low/60 backdrop-blur-sm group" style={{ borderRadius: '12px' }}>
                  <summary className="p-md cursor-pointer flex items-center justify-between list-none">
                    <div className="flex items-center gap-2"><span className="material-symbols-outlined text-primary text-[22px]">analytics</span><span className="font-label-md text-on-surface text-[16px]">Evidence &amp; Decision</span></div>
                    <span className="material-symbols-outlined text-on-surface-variant transition-transform group-open:rotate-180">expand_more</span>
                  </summary>
                  <div className="p-md border-t border-surface-container-high flex flex-col gap-md">
                    <div>
                      <div className="font-label-sm text-outline uppercase tracking-wider mb-2">Understood request</div>
                      <div className="flex flex-wrap gap-2">
                        {(!safe.length && !soft.length) ? <span className="font-label-sm text-on-surface-variant">(no special constraints)</span> : null}
                        {safe.map((c) => <span key={c} className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-success/10 text-success font-label-sm"><span className="material-symbols-outlined text-[14px]">verified_user</span>{c}</span>)}
                        {soft.map((c) => <span key={c} className="px-2 py-1 rounded-full bg-surface-container text-on-surface-variant font-label-sm">{c}</span>)}
                      </div>
                    </div>
                    <div className="grid md:grid-cols-2 gap-x-lg gap-y-2 font-label-sm text-on-surface-variant">
                      <div className="flex justify-between border-b border-surface-container-high/60 pb-1"><span>Source</span><span className="font-bold text-on-surface">{na(res.candidate_source)}</span></div>
                      <div className="flex justify-between border-b border-surface-container-high/60 pb-1"><span>Retrieval rank</span><span className="font-bold text-on-surface">{na(res.selected_rank)}</span></div>
                      <div className="flex justify-between border-b border-surface-container-high/60 pb-1"><span>Request constraints</span><span className="font-bold text-on-surface">{res.extraction_source === 'human_confirmed' ? 'Human confirmed' : 'Model extracted'}</span></div>
                      <div className="flex justify-between border-b border-surface-container-high/60 pb-1"><span>Rejected by gate</span><span className="font-bold text-error">{na(ev.rejected_by_gate)}</span></div>
                      <div className="flex justify-between border-b border-surface-container-high/60 pb-1"><span>Issues repaired</span><span className="font-bold text-on-surface">{na(ev.validation_issues_repaired)}</span></div>
                      <div className="flex justify-between"><span>Processing time</span><span className="font-bold text-on-surface">{(ev.timings_seconds && ev.timings_seconds.total != null) ? ev.timings_seconds.total + 's' : 'N/A'}</span></div>
                    </div>
                  </div>
                </details>
              </main>
            </div>
            </div>
          </div>
        </div>
      ) : null}
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
  const [files, setFiles] = useState([])
  const [previews, setPreviews] = useState([])
  const send = () => {
    if (busy) return
    if (!text.trim() && !files.length) return
    onSend(text.trim(), files, previews)
    setText(''); setFiles([]); setPreviews([])
  }
  const pick = (e) => {
    const selected = Array.from(e.target.files || [])
    if (selected.length) {
      setFiles((prev) => [...prev, ...selected])
      setPreviews((prev) => [...prev, ...selected.map((f) => URL.createObjectURL(f))])
    }
    e.target.value = ''
  }
  const remove = (index) => {
    setFiles((prev) => prev.filter((_, i) => i !== index))
    setPreviews((prev) => prev.filter((_, i) => i !== index))
  }
  return (
    <div className="fixed bottom-0 left-0 w-full z-50 flex flex-col items-center px-md pb-lg pt-md bg-gradient-to-t from-background via-background to-transparent pointer-events-none">
      {previews.length ? (
        <div className="w-full max-w-[680px] mb-2 flex items-center gap-2 pointer-events-auto">
          <div className="flex flex-wrap gap-2">
            {previews.map((preview, i) => (
              <div key={preview} className="relative h-12 w-12 rounded-lg overflow-hidden border border-outline-variant shadow-low">
                <img src={preview} className="h-full w-full object-cover" alt="" />
                <button onClick={() => remove(i)} className="absolute top-0 right-0 w-5 h-5 bg-error text-on-primary rounded-bl flex items-center justify-center">
                  <span className="material-symbols-outlined text-[14px]">close</span>
                </button>
              </div>
            ))}
          </div>
          <div className="font-label-sm text-on-surface-variant">{files.length} image{files.length > 1 ? 's' : ''} selected</div>
        </div>
      ) : null}
      <div className="w-full max-w-[680px] bg-surface-container-lowest rounded-full shadow-high border border-surface-container-high flex items-center p-2 pointer-events-auto transition-shadow focus-within:shadow-md focus-within:border-primary/50">
        <label className="w-10 h-10 flex items-center justify-center text-on-surface-variant hover:bg-surface-container rounded-full transition-colors shrink-0 cursor-pointer" title="Attach an ingredient photo">
          <span className="material-symbols-outlined text-[24px]">attach_file</span>
          <input type="file" accept="image/*" multiple className="hidden" onChange={pick} />
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
