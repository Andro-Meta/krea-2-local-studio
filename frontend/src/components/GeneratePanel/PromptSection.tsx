import React, { useEffect, useState } from 'react'
import { Alert, Box, Button, Checkbox, CircularProgress, Collapse, FormControlLabel, IconButton, MenuItem, Paper, Slider, Snackbar, Stack, TextField, Tooltip, Typography } from '@mui/material'
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh'
import TipsAndUpdatesIcon from '@mui/icons-material/TipsAndUpdates'
import { useStore } from '../../store'
import { apiFetch, type PromptPlan } from '../../api'
import CreatePromptFromImage from '../CreatePromptFromImage'

const ABLITERATED_QWEN = 'huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated'

function wandChoiceFromModel(modelId: string) {
  if (!modelId) return 'default'
  return /Huihui-Qwen3-VL-4B-Instruct-abliterated|qwen3_vl_4b_abliterated/i.test(modelId) ? 'abliterated' : 'custom'
}

function modelFromWandChoice(choice: string, current: string) {
  if (choice === 'default') return ''
  if (choice === 'abliterated') return ABLITERATED_QWEN
  return current && wandChoiceFromModel(current) === 'custom' ? current : 'custom/repo-or-path'
}

export default function PromptSection() {
  const { params, setParam, setLoras } = useStore()
  const [expanding, setExpanding] = useState(false)
  const [planning, setPlanning] = useState(false)
  const [xperimenting, setXperimenting] = useState(false)
  const [plan, setPlan] = useState<PromptPlan | null>(null)
  const [notice, setNotice] = useState<{ message: string; severity: 'success' | 'warning' | 'error' } | null>(null)
  const [wandModel, setWandModel] = useState('')

  useEffect(() => {
    apiFetch.settings()
      .then(settings => setWandModel(settings.local_qwen_model_id ?? ''))
      .catch(() => undefined)
  }, [])

  const handleExpand = async () => {
    if (!params.prompt || expanding) return
    setExpanding(true)
    try {
      await apiFetch.updateSettings({
        prompt_expander_backend: 'local',
        local_llm_backend: 'transformers',
        local_qwen_model_id: wandModel,
      })
      const { expanded, changed, error, backend } = await apiFetch.expandPrompt(params.prompt)
      if (changed && expanded) {
        setParam('prompt', expanded)
        const label = wandChoiceFromModel(wandModel) === 'abliterated'
          ? 'Abliterated Qwen3-VL'
          : backend === 'openrouter' ? 'OpenRouter' : backend === 'ideogram-json' ? 'Ideogram JSON' : 'Local Qwen3-VL'
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

  const handleXperiment = async () => {
    if (xperimenting) return
    setXperimenting(true)
    try {
      const result = await apiFetch.setupXperiment()
      setWandModel(result.local_qwen_model_id ?? ABLITERATED_QWEN)
      apiFetch.loras().then(setLoras).catch(() => undefined)
      const xperimentLoras = (result.loras?.length ? result.loras : [result.lora]).map(lora => ({
        name: lora.name,
        filename: lora.filename,
        strength: lora.strength,
        enabled: true,
        block_filter: lora.block_filter ?? (lora.name === 'Krea2-realism-V1' ? 'late' : 'style_safe'),
      }))
      const xperimentLoraNames = new Set(xperimentLoras.map(lora => lora.name))
      setParam('diffusion_engine', 'native_pytorch')
      setParam('model_profile', 'krea_turbo')
      setParam('checkpoint', 'turbo')
      setParam('quantization', 'fp8')
      setParam('steps', result.sampler.steps)
      setParam('cfg', result.sampler.cfg)
      setParam('mu', 1.15)
      setParam('sampler', result.sampler.sampler as typeof params.sampler)
      setParam('scheduler', result.sampler.scheduler as typeof params.scheduler)
      setParam('use_prompt_expander', result.use_prompt_expander ?? false)
      setParam('negative_prompt', '')
      setParam('loras', [
        ...params.loras.filter(lora => !xperimentLoraNames.has(lora.name)),
        ...xperimentLoras,
      ])
      const skipped = result.assets.filter(asset => asset.skipped).length
      const notes = [result.benchmark_note, ...result.warnings].filter(Boolean).join(' ')
      setNotice({
        severity: 'success',
        message: `Xperiment Settings applied. ${skipped}/${result.assets.length} assets were already installed.${notes ? ` Notes: ${notes}` : ''}`,
      })
    } catch (err: any) {
      const status = err?.response?.status
      const detail = err?.response?.data?.detail
      setNotice({
        severity: 'error',
        message: status === 405
          ? 'Xperiment setup route is stale. Restart run.bat so the backend and browser bundle both use the latest code.'
          : detail ?? err.message ?? 'Xperiment setup failed.',
      })
    } finally {
      setXperimenting(false)
    }
  }

  return (
    <Stack spacing={1}>
      <Paper variant="outlined" sx={{ p: 1.25, bgcolor: 'background.default' }}>
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} alignItems={{ xs: 'stretch', sm: 'center' }} justifyContent="space-between">
          <Box>
            <Typography variant="subtitle2">Xperiment Settings</Typography>
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
              One-click Krea2 Turbo setup: assets, Wan VAE override, 6 steps, CFG 0, beta57, and Realism LoKr late@0.55.
            </Typography>
          </Box>
          <Button
            type="button"
            variant="outlined"
            size="small"
            onClick={handleXperiment}
            disabled={xperimenting}
            startIcon={xperimenting ? <CircularProgress size={14} color="inherit" /> : undefined}
          >
            {xperimenting ? 'Setting up...' : 'Apply Xperiment'}
          </Button>
        </Stack>
      </Paper>
      <Stack spacing={1}>
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
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} alignItems={{ xs: 'stretch', sm: 'center' }}>
          <CreatePromptFromImage
            value={params.prompt}
            onChange={prompt => setParam('prompt', prompt)}
            compact
          />
          <TextField
            select
            label="Wand model"
            value={wandChoiceFromModel(wandModel)}
            onChange={e => setWandModel(modelFromWandChoice(e.target.value, wandModel))}
            size="small"
            sx={{ minWidth: { xs: '100%', sm: 260 } }}
            helperText="Controls the Magic wand prompt expander."
          >
            <MenuItem value="default">Default Qwen3-VL</MenuItem>
            <MenuItem value="abliterated">Abliterated Qwen3-VL</MenuItem>
            <MenuItem value="custom">Custom repo/path</MenuItem>
          </TextField>
          {wandChoiceFromModel(wandModel) === 'custom' && (
            <TextField
              label="Custom wand model"
              value={wandModel}
              onChange={e => setWandModel(e.target.value)}
              size="small"
              sx={{ minWidth: { xs: '100%', sm: 260 } }}
            />
          )}
          <Button
            variant="outlined"
            onClick={handleExpand}
            disabled={expanding || !params.prompt}
            startIcon={expanding ? <CircularProgress size={16} color="inherit" /> : <AutoFixHighIcon />}
            sx={{ alignSelf: { xs: 'stretch', sm: 'flex-start' }, minHeight: 40 }}
          >
            Magic wand
          </Button>
          <Tooltip
            title={
              <span>
                <b>Krea 2 prompting tips:</b><br />
                Use natural language; describe the scene as to a person.<br />
                Long, detailed prompts work best. Put requested text in quotes.
              </span>
            }
          >
            <Button variant="text" size="small" startIcon={<TipsAndUpdatesIcon fontSize="small" />}>
              Prompt tips
            </Button>
          </Tooltip>
        </Stack>
      </Stack>
      <Paper variant="outlined" sx={{ p: 1.5, bgcolor: 'background.default' }}>
        <Stack spacing={1}>
          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} alignItems={{ sm: 'center' }}>
            <FormControlLabel
              control={<Checkbox checked={params.use_prompt_planner} onChange={e => setParam('use_prompt_planner', e.target.checked)} />}
              label="Enhance for adherence"
            />
            <FormControlLabel
              control={<Checkbox checked={params.use_prompt_expander} onChange={e => setParam('use_prompt_expander', e.target.checked)} />}
              label="Expand on generate"
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
      <Paper variant="outlined" sx={{ p: 1.5, bgcolor: 'background.default' }}>
        <Stack spacing={1}>
          <FormControlLabel
            control={<Checkbox checked={params.think_steering_enabled} onChange={e => setParam('think_steering_enabled', e.target.checked)} />}
            label="Expression steering (<think> block)"
          />
          <Typography variant="caption" color="text.secondary" sx={{ px: 1, mt: -0.5 }}>
            Restores emotion/intensity that Turbo's distillation flattens, in-distribution — a gentler alternative to the Emotion rebalance preset. Leave the text blank to use the default expression nudge.
          </Typography>
          <Collapse in={params.think_steering_enabled}>
            <TextField
              label="Think text (optional)"
              value={params.think_text}
              onChange={e => setParam('think_text', e.target.value)}
              multiline
              minRows={2}
              maxRows={4}
              fullWidth
              size="small"
              placeholder="e.g. show genuine fear and tension, dramatic lighting…"
            />
          </Collapse>
        </Stack>
      </Paper>
      {(params.mode !== 'txt2img' || params.checkpoint === 'raw' || params.cfg > 0) && (
        <TextField
          label="Negative prompt"
          multiline
          minRows={1}
          maxRows={3}
          fullWidth
          value={params.negative_prompt}
          onChange={e => setParam('negative_prompt', e.target.value)}
          placeholder={params.mode === 'txt2img' ? 'Optional for RAW / CFG; Turbo usually leaves this empty…' : 'What to avoid…'}
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
