import React from 'react'
import { Accordion, AccordionDetails, AccordionSummary, Box, Button, Checkbox, FormControlLabel, IconButton, Paper, Slider, Stack, TextField, Typography } from '@mui/material'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import DeleteIcon from '@mui/icons-material/Delete'
import { useStore, type RegionalPrompt } from '../../store'

const emptyRegion = (): RegionalPrompt => ({
  prompt: '',
  negative_prompt: '',
  mask_b64: '',
  strength: 1,
  feather: 24,
  normalize: true,
  visible: true,
  lora_filter: '',
})

function readAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result || ''))
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

export default function AdvancedSceneSection() {
  const { params, setParam } = useStore()
  const regions = params.regional_prompts

  const updateRegion = (index: number, patch: Partial<RegionalPrompt>) => {
    setParam('regional_prompts', regions.map((region, i) => i === index ? { ...region, ...patch } : region))
  }

  const addRegion = () => {
    if (regions.length >= 8) return
    setParam('regional_prompts', [...regions, emptyRegion()])
  }

  const removeRegion = (index: number) => {
    setParam('regional_prompts', regions.filter((_, i) => i !== index))
  }

  const addMask = async (index: number, file?: File) => {
    if (!file) return
    updateRegion(index, { mask_b64: await readAsDataUrl(file) })
  }

  return (
    <Accordion disableGutters>
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Typography variant="subtitle2">Advanced Scene Regions</Typography>
      </AccordionSummary>
      <AccordionDetails>
        <Stack spacing={1.5}>
          <Typography variant="body2" color="text.secondary">
            Add mask-guided regional instructions. Defaults normalize overlaps and keep the base prompt at 30% influence.
          </Typography>
          <FormControlLabel
            control={<Checkbox checked={params.regional_normalize_masks} onChange={e => setParam('regional_normalize_masks', e.target.checked)} />}
            label="Normalize overlapping masks"
          />
          <Box>
            <Typography variant="caption" color="text.secondary">Base prompt strength: {params.regional_base_prompt_strength.toFixed(2)}</Typography>
            <Slider
              min={0}
              max={1}
              step={0.05}
              value={params.regional_base_prompt_strength}
              onChange={(_, value) => setParam('regional_base_prompt_strength', value as number)}
            />
          </Box>
          {regions.map((region, index) => (
            <Paper key={index} variant="outlined" sx={{ p: 1.5 }}>
              <Stack spacing={1}>
                <Stack direction="row" justifyContent="space-between" alignItems="center">
                  <Typography variant="subtitle2">Region {index + 1}</Typography>
                  <IconButton size="small" onClick={() => removeRegion(index)}><DeleteIcon fontSize="small" /></IconButton>
                </Stack>
                <TextField label="Region prompt" value={region.prompt} onChange={e => updateRegion(index, { prompt: e.target.value })} size="small" fullWidth />
                <TextField label="Region negative prompt" value={region.negative_prompt} onChange={e => updateRegion(index, { negative_prompt: e.target.value })} size="small" fullWidth />
                <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} alignItems={{ sm: 'center' }}>
                  <Button variant="outlined" component="label" size="small">
                    {region.mask_b64 ? 'Mask set' : 'Add mask'}
                    <input hidden type="file" accept="image/*" onChange={e => void addMask(index, e.target.files?.[0])} />
                  </Button>
                  <FormControlLabel control={<Checkbox checked={region.visible} onChange={e => updateRegion(index, { visible: e.target.checked })} />} label="Visible" />
                  <FormControlLabel control={<Checkbox checked={region.normalize} onChange={e => updateRegion(index, { normalize: e.target.checked })} />} label="Normalize" />
                  <TextField label="LoRA filter" value={region.lora_filter} onChange={e => updateRegion(index, { lora_filter: e.target.value })} size="small" />
                </Stack>
                <Box>
                  <Typography variant="caption" color="text.secondary">Strength: {region.strength.toFixed(2)}</Typography>
                  <Slider min={0} max={2} step={0.05} value={region.strength} onChange={(_, value) => updateRegion(index, { strength: value as number })} />
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">Feather: {region.feather}px</Typography>
                  <Slider min={0} max={128} step={4} value={region.feather} onChange={(_, value) => updateRegion(index, { feather: value as number })} />
                </Box>
              </Stack>
            </Paper>
          ))}
          <Button onClick={addRegion} disabled={regions.length >= 8} variant="outlined">Add region</Button>
        </Stack>
      </AccordionDetails>
    </Accordion>
  )
}
