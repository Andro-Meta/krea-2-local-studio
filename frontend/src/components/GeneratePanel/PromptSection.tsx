import React, { useState } from 'react'
import { Alert, Box, Button, Checkbox, CircularProgress, Collapse, FormControlLabel, IconButton, Paper, Slider, Snackbar, Stack, TextField, Tooltip, Typography } from '@mui/material'
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh'
import { useStore } from '../../store'
import { apiFetch, type PromptPlan } from '../../api'
import CreatePromptFromImage from '../CreatePromptFromImage'

export default function PromptSection() {
  const { params, setParam } = useStore()
  const [expanding, setExpanding] = useState(false)
  const [planning, setPlanning] = useState(false)
  const [plan, setPlan] = useState<PromptPlan | null>(null)
  const [notice, setNotice] = useState<{ message: string; severity: 'success' | 'warning' | 'error' } | null>(null)

  const handleExpand = async () => {
    if (!params.prompt || expanding) return
    setExpanding(true)
    try {
      const { expanded, changed, error, backend } = await apiFetch.expandPrompt(params.prompt)
      if (changed && expanded) {
        setParam('prompt', expanded)
        const label = backend === 'openrouter' ? 'OpenRouter' : backend === 'ideogram-json' ? 'Ideogram JSON' : 'Local Qwen3-VL'
        setNotice({ severity: 'success', message: `Prompt expanded with ${label}.` })
      } else if (error) {
        setNotice({ severity: 'warning', message: error })
      } else {
        setNotice({ severity: 'warning', message: 'The wand did not return a different prompt.' })
      }
    } catch (err) {
      setNotice({ severity: 'error', message: err instanceof Error ? err.message : 'Prompt expansion failed.' })
    } finally {
      setExpanding(false)
    }
  }

  const handlePlan = async () => {
    if (!params.prompt || planning) return
    setPlanning(true)
    try {
      const result = await apiFetch.planPrompt(params.prompt, params.prompt_planner_max_tokens)
      setPlan(result)
      setParam('prompt_planner_show_output', true)
      if (result.error) {
        setNotice({ severity: 'warning', message: result.error })
      } else {
        const label = result.backend === 'local' ? 'Local Qwen3-VL' : 'heuristic fallback'
        setNotice({ severity: 'success', message: `Prompt planned with ${label}.` })
      }
    } catch (err) {
      setNotice({ severity: 'error', message: err instanceof Error ? err.message : 'Prompt planning failed.' })
    } finally {
      setPlanning(false)
    }
  }

  return (
    <Stack spacing={1}>
      <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
        <TextField
          label="Prompt"
          multiline
          minRows={3}
          maxRows={8}
          fullWidth
          value={params.prompt}
          onChange={e => setParam('prompt', e.target.value)}
          placeholder="Describe the image you want to create…"
        />
        <Tooltip title="Expand prompt with AI">
          <span>
            <IconButton onClick={handleExpand} disabled={expanding || !params.prompt} sx={{ mt: 0.5 }}>
              {expanding ? <CircularProgress size={20} /> : <AutoFixHighIcon />}
            </IconButton>
          </span>
        </Tooltip>
      </Box>
      <CreatePromptFromImage
        value={params.prompt}
        onChange={prompt => setParam('prompt', prompt)}
        compact
      />
      <Paper variant="outlined" sx={{ p: 1.5, bgcolor: 'background.default' }}>
        <Stack spacing={1}>
          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} alignItems={{ sm: 'center' }}>
            <FormControlLabel
              control={<Checkbox checked={params.use_prompt_planner} onChange={e => setParam('use_prompt_planner', e.target.checked)} />}
              label="Enhance for adherence"
            />
            <FormControlLabel
              control={<Checkbox checked={params.prompt_planner_lock_original} onChange={e => setParam('prompt_planner_lock_original', e.target.checked)} />}
              label="Lock original prompt"
            />
            <FormControlLabel
              control={<Checkbox checked={params.prompt_planner_use_regions} onChange={e => setParam('prompt_planner_use_regions', e.target.checked)} />}
              label="Use planner for regions"
            />
            <Button size="small" variant="outlined" onClick={handlePlan} disabled={planning || !params.prompt}>
              {planning ? 'Planning…' : 'Show planner output'}
            </Button>
          </Stack>
          <Box sx={{ px: 1 }}>
            <Typography variant="caption" color="text.secondary">Max planner tokens: {params.prompt_planner_max_tokens}</Typography>
            <Slider
              min={128}
              max={1600}
              step={64}
              size="small"
              value={params.prompt_planner_max_tokens}
              onChange={(_, value) => setParam('prompt_planner_max_tokens', value as number)}
            />
          </Box>
          <Collapse in={params.prompt_planner_show_output && !!plan}>
            {plan ? (
              <Stack spacing={1}>
                <TextField label="Planned prompt" value={plan.planned_prompt} multiline minRows={3} size="small" fullWidth InputProps={{ readOnly: true }} />
                {plan.negative_prompt ? <TextField label="Planner negative prompt" value={plan.negative_prompt} size="small" fullWidth InputProps={{ readOnly: true }} /> : null}
                <Typography variant="caption" color="text.secondary">
                  Subject: {plan.subject || 'n/a'} · Composition: {plan.composition || 'n/a'} · Lighting: {plan.lighting || 'n/a'}
                </Typography>
                <Stack direction="row" spacing={1}>
                  <Button size="small" variant="contained" onClick={() => {
                    setParam('prompt', plan.planned_prompt)
                    if (plan.negative_prompt && !params.negative_prompt) setParam('negative_prompt', plan.negative_prompt)
                  }}>
                    Apply planned prompt
                  </Button>
                  <Button size="small" onClick={() => setParam('prompt_planner_show_output', false)}>Hide</Button>
                </Stack>
              </Stack>
            ) : null}
          </Collapse>
        </Stack>
      </Paper>
      {params.mode !== 'txt2img' && (
        <TextField
          label="Negative prompt"
          multiline
          minRows={1}
          maxRows={3}
          fullWidth
          value={params.negative_prompt}
          onChange={e => setParam('negative_prompt', e.target.value)}
          placeholder="What to avoid…"
          size="small"
        />
      )}
      <Snackbar
        open={!!notice}
        autoHideDuration={5000}
        onClose={() => setNotice(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        {notice ? (
          <Alert severity={notice.severity} variant="filled" onClose={() => setNotice(null)}>
            {notice.message}
          </Alert>
        ) : undefined}
      </Snackbar>
    </Stack>
  )
}
