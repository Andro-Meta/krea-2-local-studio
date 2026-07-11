import React, { useEffect, useState } from 'react'
import { Alert, Box, Button, Checkbox, Chip, CircularProgress, Collapse, FormControlLabel, MenuItem, Paper, Slider, Snackbar, Stack, TextField, Tooltip, Typography } from '@mui/material'
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh'
import TipsAndUpdatesIcon from '@mui/icons-material/TipsAndUpdates'
import { useStore } from '../../store'
import { apiFetch, type MoodboardSuggestion, type PromptPlan } from '../../api'
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
  const { params, setParam, setParams, setPromptBusy, moodboardSuggestions, setMoodboardSuggestions } = useStore()
  const [expanding, setExpanding] = useState(false)
  const [planning, setPlanning] = useState(false)
  const [plan, setPlan] = useState<PromptPlan | null>(null)
  const [notice, setNotice] = useState<{ message: string; severity: 'success' | 'warning' | 'error' } | null>(null)
  const [preWandPrompt, setPreWandPrompt] = useState('')
  const [wandModel, setWandModel] = useState('')
  const [wandBackend, setWandBackend] = useState<'local' | 'openrouter' | 'ideogram-json'>('local')
  const [localLlmBackend, setLocalLlmBackend] = useState<'comfy' | 'transformers' | 'gguf_server'>('comfy')
  const [wandDevice, setWandDevice] = useState<'auto' | 'cuda' | 'cpu'>('auto')
  const [showWandAdvanced, setShowWandAdvanced] = useState(false)
  const [showPromptTools, setShowPromptTools] = useState(false)
  // Snapshot of server-side wand settings so we only PUT /api/settings (an
  // admin-only endpoint) when the user actually changed something here.
  const [serverWand, setServerWand] = useState<{ model: string; backend: string; llm: string; device: string } | null>(null)

  useEffect(() => {
    apiFetch.settings()
      .then(settings => {
        setWandModel(settings.local_qwen_model_id ?? '')
        setWandBackend(settings.prompt_expander_backend ?? 'local')
        setLocalLlmBackend(settings.local_llm_backend ?? 'comfy')
        setWandDevice(settings.local_qwen_device ?? 'auto')
        setServerWand({
          model: settings.local_qwen_model_id ?? '',
          backend: settings.prompt_expander_backend ?? 'local',
          llm: settings.local_llm_backend ?? 'comfy',
          device: settings.local_qwen_device ?? 'auto',
        })
      })
      .catch(() => undefined)
  }, [])

  const handleExpand = async () => {
    if (!params.prompt || expanding) return
    const originalPrompt = params.prompt
    setExpanding(true)
    setPromptBusy(true)
    try {
      const wandChanged = !serverWand
        || serverWand.model !== wandModel || serverWand.backend !== wandBackend
        || serverWand.llm !== localLlmBackend || serverWand.device !== wandDevice
      if (wandChanged) {
        // Settings PUT is admin-only in share mode. Non-admins still get the
        // wand: the chosen backend rides along in the expand request instead.
        try {
          await apiFetch.updateSettings({
            prompt_expander_backend: wandBackend,
            local_llm_backend: localLlmBackend,
            local_qwen_model_id: wandModel,
            local_qwen_device: wandDevice,
          })
          setServerWand({ model: wandModel, backend: wandBackend, llm: localLlmBackend, device: wandDevice })
        } catch {
          setNotice({ severity: 'warning', message: 'Wand settings are admin-only; using the server defaults for this expansion.' })
        }
      }
      const { expanded, changed, error, backend, suggested_moodboards } = await apiFetch.expandPrompt(params.prompt, wandBackend)
      setMoodboardSuggestions(suggested_moodboards ?? [])
      if (changed && expanded) {
        setPreWandPrompt(originalPrompt)
        setParam('prompt', expanded)
        const label = wandBackend === 'local' && localLlmBackend === 'transformers' && wandChoiceFromModel(wandModel) === 'abliterated'
          ? 'Abliterated Qwen3-VL'
          : backend === 'openrouter' ? 'OpenRouter' : backend === 'ideogram-json' ? 'Ideogram JSON' : localLlmBackend === 'gguf_server' ? 'local GGUF helper' : localLlmBackend === 'comfy' ? 'ComfyUI QwenVL' : 'Local Qwen3-VL'
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
      setPromptBusy(false)
    }
  }

  const undoMagicWand = () => {
    if (!preWandPrompt) return
    setParam('prompt', preWandPrompt)
    setPreWandPrompt('')
    setNotice({ severity: 'success', message: 'Restored the prompt from before Magic Wand.' })
  }

  const applyMoodboardSuggestion = (board: MoodboardSuggestion) => {
    const ids = Array.from(new Set([...params.selected_moodboard_ids, board.id]))
    const uuids = board.uuid ? Array.from(new Set([...params.moodboard_uuids, board.uuid])) : params.moodboard_uuids
    setParams({
      selected_moodboard_ids: ids,
      moodboard_uuids: uuids,
      moodboard_strength: params.moodboard_strength || 0.35,
    })
    setNotice({ severity: 'success', message: `Applied moodboard "${board.title}".` })
  }

  const removeMoodboard = (id: number, uuid = '') => {
    setParams({
      selected_moodboard_ids: params.selected_moodboard_ids.filter(existing => existing !== id),
      moodboard_uuids: uuid ? params.moodboard_uuids.filter(existing => existing !== uuid) : params.moodboard_uuids,
    })
  }

  const activeMoodboards = params.selected_moodboard_ids.map(id => {
    const suggestion = moodboardSuggestions.find(board => board.id === id)
    return { id, title: suggestion?.title ?? `Moodboard #${id}`, uuid: suggestion?.uuid ?? '' }
  })

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

  // Xperiment setup lives in QuickPresets (the "Xperiment" chip); the old
  // duplicate handler here was never wired to any UI and has been removed.

  return (
    <Stack spacing={1}>
      <Paper variant="outlined" sx={{ p: 1.25, bgcolor: 'background.default' }}>
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} alignItems={{ xs: 'stretch', sm: 'center' }} justifyContent="space-between">
          <Box>
            <Typography variant="subtitle2">Create prompt from image</Typography>
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
              Upload an image and the local Qwen3-VL reverse-engineers a prompt to recreate it. Add optional guidance to focus on — or change — specific parts; leave it blank for a full auto prompt.
            </Typography>
          </Box>
          <CreatePromptFromImage value={params.prompt} onChange={prompt => setParam('prompt', prompt)} label="Create from image" withGuidance />
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
        {activeMoodboards.length > 0 && (
          <Paper variant="outlined" sx={{ p: 1, borderColor: 'rgba(208,188,255,0.35)', bgcolor: 'rgba(208,188,255,0.06)' }}>
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.75 }}>
              Active moodboards are applied as text guidance on generate:
            </Typography>
            <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
              {activeMoodboards.map(board => (
                <Chip
                  key={board.id}
                  size="small"
                  color="primary"
                  label={board.title}
                  onDelete={() => removeMoodboard(board.id, board.uuid)}
                />
              ))}
            </Stack>
          </Paper>
        )}
        <Stack direction={{ xs: 'column', md: 'row' }} spacing={1} alignItems={{ xs: 'stretch', md: 'flex-start' }}>
          <Button
            variant="outlined"
            onClick={handleExpand}
            disabled={expanding || !params.prompt}
            startIcon={expanding ? <CircularProgress size={16} color="inherit" /> : <AutoFixHighIcon />}
            sx={{ alignSelf: { xs: 'stretch', md: 'flex-start' }, minHeight: 40 }}
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
            <Button variant="text" size="small" startIcon={<TipsAndUpdatesIcon fontSize="small" />} sx={{ alignSelf: { xs: 'stretch', md: 'flex-start' }, minHeight: 40 }}>
              Prompt tips
            </Button>
          </Tooltip>
          {preWandPrompt && (
            <Button
              variant="text"
              size="small"
              onClick={undoMagicWand}
              sx={{ alignSelf: { xs: 'stretch', md: 'flex-start' }, minHeight: 40 }}
            >
              Undo Magic Wand
            </Button>
          )}
        </Stack>
        <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mt: -0.25 }}>
          Magic Wand expands your prompt and can suggest matching moodboards. Applying a suggestion makes it visible above.
        </Typography>
        {moodboardSuggestions.length > 0 && (
          <Paper variant="outlined" sx={{ p: 1, borderColor: 'rgba(202,196,208,0.18)', bgcolor: 'rgba(255,255,255,0.03)' }}>
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.75 }}>
              Magic Wand found moodboards that fit this prompt. Apply one to add its text guidance; image influence remains opt-in in Image Prompt.
            </Typography>
            <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
              {moodboardSuggestions.map(board => (
                <Tooltip key={board.id} title={board.reason || 'Apply this moodboard'} arrow>
                  <Chip
                    clickable
                    size="small"
                    label={board.title}
                    color={params.selected_moodboard_ids.includes(board.id) ? 'primary' : 'default'}
                    variant={params.selected_moodboard_ids.includes(board.id) ? 'filled' : 'outlined'}
                    onClick={() => applyMoodboardSuggestion(board)}
                  />
                </Tooltip>
              ))}
            </Stack>
          </Paper>
        )}
      </Stack>
      <Box>
        <Button size="small" variant="text" onClick={() => setShowPromptTools(v => !v)} sx={{ minHeight: 32 }}>
          {showPromptTools ? 'Hide prompt tools' : 'Prompt tools — planner, expression steering & negative prompt'}
        </Button>
        <Collapse in={showPromptTools}>
          <Stack spacing={1} sx={{ pt: 0.5 }}>
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
      <TextField
        label="Negative prompt"
        multiline
        minRows={1}
        maxRows={3}
        fullWidth
        value={params.negative_prompt}
        onChange={e => setParam('negative_prompt', e.target.value)}
        placeholder="What to avoid (optional; Turbo usually leaves this empty)…"
        size="small"
      />
          </Stack>
        </Collapse>
      </Box>
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
