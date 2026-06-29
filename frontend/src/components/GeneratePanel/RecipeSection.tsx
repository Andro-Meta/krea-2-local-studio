import React, { useEffect, useState } from 'react'
import { Accordion, AccordionDetails, AccordionSummary, Alert, Button, IconButton, MenuItem, Stack, TextField, Tooltip, Typography } from '@mui/material'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import DeleteIcon from '@mui/icons-material/Delete'
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined'
import { apiFetch, type PromptRecipe } from '../../api'
import { useStore } from '../../store'

export default function RecipeSection() {
  const { params, setParam, setParams } = useStore()
  const [recipes, setRecipes] = useState<PromptRecipe[]>([])
  const [selected, setSelected] = useState('')
  const [name, setName] = useState('')
  const [notice, setNotice] = useState('')

  const refresh = () => {
    apiFetch.promptRecipes().then(r => setRecipes(r.items)).catch(() => setRecipes([]))
  }

  useEffect(refresh, [])

  const save = async () => {
    const recipeName = name.trim() || params.prompt.slice(0, 48) || 'Untitled recipe'
    const recipe = await apiFetch.savePromptRecipe({
      name: recipeName,
      prompt: params.prompt,
      negative_prompt: params.negative_prompt,
      loras: params.loras,
      mood: params.mood,
      moodboard_strength: params.moodboard_strength,
      moodboard_ids: params.selected_moodboard_ids,
      moodboard_uuids: params.moodboard_uuids,
      style_references: params.style_references,
      regional_prompts: params.regional_prompts,
      seed_variance_preset: params.seed_variance_preset,
      krea_enhancer_variant: params.krea_enhancer_variant,
      rebalance_preset: params.rebalance_preset,
    })
    setSelected(recipe.id)
    setName('')
    setNotice(`Saved recipe "${recipe.name}".`)
    refresh()
  }

  const apply = () => {
    const recipe = recipes.find(item => item.id === selected)
    if (!recipe) return
    setParam('prompt', recipe.prompt)
    setParam('negative_prompt', recipe.negative_prompt)
    setParam('loras', recipe.loras as any)
    setParam('mood', recipe.mood || '')
    setParam('moodboard_strength', recipe.moodboard_strength ?? 0.35)
    setParam('selected_moodboard_ids', recipe.moodboard_ids)
    setParam('moodboard_uuids', recipe.moodboard_uuids)
    setParam('style_references', recipe.style_references as any)
    setParam('regional_prompts', recipe.regional_prompts as any)
    setParam('seed_variance_preset', recipe.seed_variance_preset as any)
    setParam('krea_enhancer_variant', recipe.krea_enhancer_variant as any)
    setParam('krea_enhancer_enabled', recipe.krea_enhancer_variant !== 'off')
    setParams({
      rebalance_preset: recipe.rebalance_preset as any,
      rebalance_mode: recipe.rebalance_preset === 'legacy' ? 'legacy_multiply' : 'rms_renorm',
      rebalance_renormalize: recipe.rebalance_preset !== 'legacy',
      rebalance_multiplier: recipe.rebalance_preset === 'legacy' ? 4.0 : 1.0,
    })
    setNotice(`Applied recipe "${recipe.name}".`)
  }

  const remove = async () => {
    if (!selected) return
    await apiFetch.deletePromptRecipe(selected)
    setSelected('')
    setNotice('Deleted recipe.')
    refresh()
  }

  return (
    <Accordion disableGutters>
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center' }}>
          Prompt, LoRA, Moodboard Recipes
          <Tooltip title="Recipes are reusable presets. They save the current prompt, negative prompt, LoRAs, selected moodboards, style references, regional prompts, seed variance, enhancer, and rebalance settings.">
            <InfoOutlinedIcon sx={{ fontSize: 15, color: 'text.disabled', ml: 0.75 }} />
          </Tooltip>
        </Typography>
      </AccordionSummary>
      <AccordionDetails>
        <Stack spacing={1.5}>
          <Alert severity="info" sx={{ py: 0.75 }}>
            Save the current setup as a reusable recipe after you have a prompt or moodboard stack you like. Apply restores the saved settings into this panel.
          </Alert>
          {notice ? <Alert severity="success" onClose={() => setNotice('')}>{notice}</Alert> : null}
          <TextField
            label="Recipe name"
            value={name}
            onChange={e => setName(e.target.value)}
            size="small"
            fullWidth
            helperText={params.prompt.trim() ? 'Optional. Empty uses the first part of your prompt.' : 'Add a prompt before saving a recipe.'}
          />
          <Tooltip title={!params.prompt.trim() ? 'Recipes need at least prompt text to save.' : 'Save the current prompt and advanced style setup.'}>
            <span>
              <Button variant="outlined" onClick={save} disabled={!params.prompt.trim()} fullWidth>
                Save current recipe
              </Button>
            </span>
          </Tooltip>
          <Stack direction="row" spacing={1} alignItems="center">
            <TextField select label="Saved recipes" value={selected} onChange={e => setSelected(e.target.value)} size="small" fullWidth>
              <MenuItem value="">{recipes.length ? 'Choose recipe' : 'No saved recipes yet'}</MenuItem>
              {recipes.map(recipe => <MenuItem key={recipe.id} value={recipe.id}>{recipe.name}</MenuItem>)}
            </TextField>
            <Tooltip title={!selected ? 'Choose a saved recipe first.' : 'Apply this saved setup to the current generation panel.'}>
              <span><Button variant="contained" onClick={apply} disabled={!selected}>Apply</Button></span>
            </Tooltip>
            <Tooltip title={!selected ? 'Choose a saved recipe first.' : 'Delete this saved recipe.'}>
              <span><IconButton onClick={remove} disabled={!selected}><DeleteIcon /></IconButton></span>
            </Tooltip>
          </Stack>
          <Typography variant="caption" color="text.secondary">
            Recipes restore prompt text, negatives, LoRAs, moodboards, style refs, regional prompts, seed variance, enhancer, and rebalance preset.
          </Typography>
        </Stack>
      </AccordionDetails>
    </Accordion>
  )
}
