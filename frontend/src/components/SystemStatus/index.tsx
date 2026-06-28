import React, { useEffect, useState } from 'react'
import { Alert, Box, Button, Chip, CircularProgress, FormControlLabel, LinearProgress, Paper, Stack, Switch, TextField, Typography } from '@mui/material'
import GpuIcon from '@mui/icons-material/Memory'
import { apiFetch, type AppSettings, type AuthSession, type KreaServerProcess, type QualityAsset, type ShareUser, type SharingStatus, type SystemReport } from '../../api'
import { useStore } from '../../store'

function GBBar({ label, used, total }: { label: string; used?: number; total?: number }) {
  const pct = (used != null && total != null && total > 0) ? (used / total * 100) : 0
  const free = (total != null && used != null) ? total - used : null
  return (
    <Box>
      <Stack direction="row" justifyContent="space-between" mb={0.25}>
        <Typography variant="caption" sx={{ color: 'text.secondary' }}>{label}</Typography>
        <Typography variant="caption" sx={{ fontFamily: 'Roboto Mono', fontSize: 11 }}>
          {free != null ? `${free.toFixed(1)} GB free` : '—'}
        </Typography>
      </Stack>
      <LinearProgress variant="determinate" value={pct} sx={{ height: 6, borderRadius: 100 }} />
    </Box>
  )
}

export default function SystemStatus() {
  const [report, setReport] = useState<SystemReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [fetchError, setFetchError] = useState('')
  const [cpPath, setCpPath] = useState('')
  const [quant, setQuant] = useState('fp8')
  const [blocksToSwap, setBlocksToSwap] = useState(0)
  const [pathTouched, setPathTouched] = useState(false)
  const [loadingModel, setLoadingModel] = useState(false)
  const [loadError, setLoadError] = useState('')
  const [downloadingSupport, setDownloadingSupport] = useState(false)
  const [supportMessage, setSupportMessage] = useState<{ severity: 'success' | 'error'; text: string } | null>(null)
  const [settings, setSettings] = useState<AppSettings | null>(null)
  const [settingsDraft, setSettingsDraft] = useState({
    prompt_expander_backend: 'local' as 'local' | 'openrouter' | 'ideogram-json',
    ideogram_api_key: '',
    hf_token: '',
    openrouter_api_key: '',
    openrouter_model: 'google/gemma-4-31b-it:free',
    openrouter_free_only: true,
  })
  const [savingSettings, setSavingSettings] = useState(false)
  const [settingsMessage, setSettingsMessage] = useState<{ severity: 'success' | 'error'; text: string } | null>(null)
  const [auth, setAuth] = useState<AuthSession | null>(null)
  const [users, setUsers] = useState<ShareUser[]>([])
  const [sharing, setSharing] = useState<SharingStatus | null>(null)
  const [sharingBusy, setSharingBusy] = useState(false)
  const [sharingMessage, setSharingMessage] = useState<{ severity: 'success' | 'error'; text: string } | null>(null)
  const [sharingAutoSaving, setSharingAutoSaving] = useState(false)
  const [userMessage, setUserMessage] = useState<{ severity: 'success' | 'error'; text: string } | null>(null)
  const [newUser, setNewUser] = useState({ username: '', password: '', role: 'user' as 'admin' | 'user' })
  const [qualityAssets, setQualityAssets] = useState<{ has_hf_token: boolean; items: QualityAsset[] } | null>(null)
  const [qualityBusy, setQualityBusy] = useState<string | null>(null)
  const [qualityMessage, setQualityMessage] = useState<{ severity: 'success' | 'error'; text: string } | null>(null)
  const [memoryBusy, setMemoryBusy] = useState<string | null>(null)
  const [memoryMessage, setMemoryMessage] = useState<{ severity: 'success' | 'error' | 'info'; text: string } | null>(null)
  const [kreaProcesses, setKreaProcesses] = useState<KreaServerProcess[]>([])
  const { setSystemReport } = useStore()
  const isAdmin = auth?.role === 'admin'

  const refresh = async () => {
    setLoading(true); setFetchError('')
    try {
      const r = await apiFetch.system()
      setReport(r)
      setSystemReport(r)
      // Prefill the checkpoint form with the auto-detected path so the user can
      // load (or recover from a failed auto-load) in one click.
      if (!pathTouched && r.model_status.auto_checkpoint) {
        setCpPath(r.model_status.auto_checkpoint)
        if (r.model_status.auto_quant) setQuant(r.model_status.auto_quant)
      }
    } catch (e: any) {
      setFetchError('Cannot reach the server on port 8200. Is it running (run.bat)?')
    } finally { setLoading(false) }
  }

  useEffect(() => { refresh() }, [])

  const loadAuth = async () => {
    try {
      const session = await apiFetch.authMe()
      setAuth(session)
      return session
    } catch {
      setAuth(null)
      return null
    }
  }

  const loadSettings = async () => {
    try {
      const s = await apiFetch.settings()
      setSettings(s)
      setSettingsDraft({
        prompt_expander_backend: s.prompt_expander_backend,
        ideogram_api_key: '',
        hf_token: '',
        openrouter_api_key: '',
        openrouter_model: s.openrouter_model,
        openrouter_free_only: s.openrouter_free_only,
      })
    } catch {
      setSettingsMessage({ severity: 'error', text: 'Could not load settings.' })
    }
  }

  const loadUsers = async () => {
    try {
      setUsers(await apiFetch.listUsers())
    } catch {
      setUserMessage({ severity: 'error', text: 'Could not load users.' })
    }
  }

  const loadSharing = async () => {
    try {
      setSharing(await apiFetch.sharingStatus())
    } catch {
      setSharingMessage({ severity: 'error', text: 'Could not load Tailscale sharing status.' })
    }
  }

  const loadQualityAssets = async () => {
    try {
      setQualityAssets(await apiFetch.qualityAssets())
    } catch {
      setQualityMessage({ severity: 'error', text: 'Could not load precision editing asset status.' })
    }
  }

  useEffect(() => {
    loadAuth().then(session => {
      if (session?.role === 'admin') {
        loadSettings()
        loadUsers()
        loadSharing()
        loadQualityAssets()
      }
    })
  }, [])

  const saveHfToken = async () => {
    const token = settingsDraft.hf_token.trim()
    if (!token) return
    setSavingSettings(true)
    setQualityMessage(null)
    try {
      await apiFetch.updateSettings({ hf_token: token })
      setSettingsDraft(d => ({ ...d, hf_token: '' }))
      await loadSettings()
      await loadQualityAssets()
      setQualityMessage({ severity: 'success', text: 'Hugging Face token saved for this server session. You can now download gated assets.' })
    } catch (e: any) {
      setQualityMessage({ severity: 'error', text: e?.response?.data?.detail ?? e.message ?? 'Could not save Hugging Face token.' })
    } finally {
      setSavingSettings(false)
    }
  }

  const downloadQualityAsset = async (assetId: string) => {
    setQualityBusy(assetId)
    setQualityMessage(null)
    try {
      await apiFetch.downloadQualityAsset(assetId)
      await loadQualityAssets()
      await refresh()
      setQualityMessage({ severity: 'success', text: 'Precision editing asset is ready.' })
    } catch (e: any) {
      setQualityMessage({ severity: 'error', text: e?.response?.data?.detail ?? e.message ?? 'Quality asset download failed.' })
    } finally {
      setQualityBusy(null)
    }
  }

  const saveMagicWandSettings = async () => {
    setSavingSettings(true)
    setSettingsMessage(null)
    try {
      await apiFetch.updateSettings({
        prompt_expander_backend: settingsDraft.prompt_expander_backend,
        ...(settingsDraft.ideogram_api_key.trim() ? { ideogram_api_key: settingsDraft.ideogram_api_key.trim() } : {}),
        openrouter_model: settingsDraft.openrouter_model,
        openrouter_free_only: settingsDraft.openrouter_free_only,
        ...(settingsDraft.openrouter_api_key.trim() ? { openrouter_api_key: settingsDraft.openrouter_api_key.trim() } : {}),
      })
      await loadSettings()
      setSettingsMessage({ severity: 'success', text: 'Magic wand settings saved.' })
    } catch (e: any) {
      setSettingsMessage({ severity: 'error', text: e?.response?.data?.detail ?? e.message ?? 'Settings update failed.' })
    } finally {
      setSavingSettings(false)
    }
  }

  const loadModel = async () => {
    if (!cpPath) return
    setLoadingModel(true); setLoadError('')
    try {
      await apiFetch.loadModel(cpPath, quant, blocksToSwap)
      await refresh()
    } catch (e: any) {
      setLoadError(e?.response?.data?.detail ?? e.message)
    } finally { setLoadingModel(false) }
  }

  const unload = async () => {
    setMemoryBusy('unload')
    setMemoryMessage(null)
    try {
      await apiFetch.unloadModelMemory()
      await refresh()
      setMemoryMessage({ severity: 'success', text: 'Model unloaded and CUDA cache cleared.' })
    } catch (e: any) {
      setMemoryMessage({ severity: 'error', text: e?.response?.data?.detail ?? e.message ?? 'Could not unload model.' })
    } finally {
      setMemoryBusy(null)
    }
  }

  const releaseTransientMemory = async () => {
    setMemoryBusy('release')
    setMemoryMessage(null)
    try {
      await apiFetch.releaseTransientMemory()
      await refresh()
      setMemoryMessage({ severity: 'success', text: 'Transient encoder/cache memory released.' })
    } catch (e: any) {
      setMemoryMessage({ severity: 'error', text: e?.response?.data?.detail ?? e.message ?? 'Could not release memory.' })
    } finally {
      setMemoryBusy(null)
    }
  }

  const loadMemoryProcesses = async () => {
    setMemoryBusy('processes')
    setMemoryMessage(null)
    try {
      const result = await apiFetch.memoryProcesses()
      setKreaProcesses(result.items)
      setMemoryMessage({
        severity: result.items.length ? 'info' : 'success',
        text: result.items.length ? `Found ${result.items.length} Krea server process${result.items.length === 1 ? '' : 'es'}.` : 'No duplicate Krea server processes found.',
      })
    } catch (e: any) {
      setMemoryMessage({ severity: 'error', text: e?.response?.data?.detail ?? e.message ?? 'Could not inspect Krea server processes.' })
    } finally {
      setMemoryBusy(null)
    }
  }

  const stopMemoryProcess = async (pid: number) => {
    if (!window.confirm(`Stop Krea server process ${pid}? Only do this for duplicate servers you are not using.`)) return
    setMemoryBusy(`stop-${pid}`)
    setMemoryMessage(null)
    try {
      await apiFetch.stopMemoryProcess(pid)
      await refresh()
      await loadMemoryProcesses()
      setMemoryMessage({ severity: 'success', text: `Stopped Krea server process ${pid}.` })
    } catch (e: any) {
      setMemoryMessage({ severity: 'error', text: e?.response?.data?.detail ?? e.message ?? `Could not stop process ${pid}.` })
    } finally {
      setMemoryBusy(null)
    }
  }

  const downloadSupportModels = async () => {
    setDownloadingSupport(true)
    setSupportMessage(null)
    try {
      await apiFetch.downloadSupportModels()
      await refresh()
      setSupportMessage({ severity: 'success', text: 'Krea conditioning assets are ready.' })
    } catch (e: any) {
      setSupportMessage({ severity: 'error', text: e?.response?.data?.detail ?? e.message ?? 'Support model download failed.' })
    } finally {
      setDownloadingSupport(false)
    }
  }

  const addUser = async () => {
    setUserMessage(null)
    try {
      const updated = await apiFetch.addUser(newUser.username.trim(), newUser.password, newUser.role)
      setUsers(updated)
      setNewUser({ username: '', password: '', role: 'user' })
      setUserMessage({ severity: 'success', text: 'User saved.' })
    } catch (e: any) {
      setUserMessage({ severity: 'error', text: e?.response?.data?.detail ?? e.message ?? 'Could not save user.' })
    }
  }

  const changeUserRole = async (username: string, role: 'admin' | 'user') => {
    setUsers(await apiFetch.setUserRole(username, role))
  }

  const removeUser = async (username: string) => {
    setUsers(await apiFetch.removeUser(username))
  }

  const resetPassword = async (username: string) => {
    const password = window.prompt(`New password for ${username}`)
    if (!password) return
    try {
      await apiFetch.resetUserPassword(username, password)
      setUserMessage({ severity: 'success', text: `Password reset for ${username}.` })
    } catch (e: any) {
      setUserMessage({ severity: 'error', text: e?.response?.data?.detail ?? e.message ?? 'Could not reset password.' })
    }
  }

  const startSharing = async () => {
    setSharingBusy(true); setSharingMessage(null)
    try {
      const result = await apiFetch.startSharing()
      await loadSharing()
      setSharingMessage({ severity: 'success', text: result.url ? `Sharing at ${result.url}` : 'Sharing started.' })
    } catch (e: any) {
      setSharingMessage({ severity: 'error', text: e?.response?.data?.detail ?? e.message ?? 'Could not start sharing.' })
    } finally {
      setSharingBusy(false)
    }
  }

  const stopSharing = async () => {
    setSharingBusy(true); setSharingMessage(null)
    try {
      await apiFetch.stopSharing()
      await loadSharing()
      setSharingMessage({ severity: 'success', text: 'Stopped the /krea Funnel route.' })
    } catch (e: any) {
      setSharingMessage({ severity: 'error', text: e?.response?.data?.detail ?? e.message ?? 'Could not stop sharing.' })
    } finally {
      setSharingBusy(false)
    }
  }

  const setAutoFunnel = async (enabled: boolean) => {
    setSharingAutoSaving(true)
    setSharingMessage(null)
    try {
      await apiFetch.updateSettings({ krea_share_auto_funnel: enabled })
      await loadSettings()
      setSharingMessage({
        severity: 'success',
        text: enabled
          ? 'run.bat will start Tailscale and the /krea Funnel automatically.'
          : 'run.bat will start local sharing controls only.',
      })
    } catch (e: any) {
      setSharingMessage({ severity: 'error', text: e?.response?.data?.detail ?? e.message ?? 'Could not save sharing startup setting.' })
    } finally {
      setSharingAutoSaving(false)
    }
  }

  return (
    <Box sx={{ p: { xs: 1.5, sm: 2 }, maxWidth: 700, mx: 'auto' }}>
      <Stack spacing={2}>
        {/* GPU info */}
        <Paper sx={{ p: 2 }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center" mb={1.5}>
            <Stack direction="row" spacing={1} alignItems="center">
              <GpuIcon sx={{ color: 'primary.main' }} />
              <Typography variant="h6">Hardware</Typography>
            </Stack>
            <Button size="small" onClick={refresh} disabled={loading}>
              {loading ? <CircularProgress size={16} /> : 'Refresh'}
            </Button>
          </Stack>
          {report ? (
            <Stack spacing={1.5}>
              <Typography variant="body2" sx={{ fontFamily: 'Roboto Mono', fontSize: 13 }}>
                {report.gpu_name ?? 'No GPU detected'}
              </Typography>
              <GBBar label="VRAM" used={report.vram_total_gb != null && report.vram_free_gb != null ? report.vram_total_gb - report.vram_free_gb : undefined} total={report.vram_total_gb} />
              <GBBar label="RAM" used={report.ram_total_gb != null && report.ram_available_gb != null ? report.ram_total_gb - report.ram_available_gb : undefined} total={report.ram_total_gb} />
              {report.disk_free_gb != null && (
                <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                  Disk free: {report.disk_free_gb.toFixed(1)} GB
                </Typography>
              )}
              {(report.gpu_process_details?.length ?? 0) > 0 ? (
                <Typography variant="caption" sx={{ color: 'warning.main' }}>
                  Other GPU processes: {report.gpu_process_details?.map(proc =>
                    `${proc.name} pid ${proc.pid}${proc.used_memory_gb != null ? ` (${proc.used_memory_gb.toFixed(1)} GB)` : ''}`,
                  ).join(', ')}
                </Typography>
              ) : report.gpu_processes.length > 0 && (
                <Typography variant="caption" sx={{ color: 'warning.main' }}>
                  Other GPU processes: {report.gpu_processes.join(', ')}
                </Typography>
              )}
            </Stack>
          ) : loading ? <CircularProgress size={24} /> : (
            <Alert severity="error" sx={{ py: 0 }}>
              {fetchError || 'No hardware data.'}
            </Alert>
          )}
        </Paper>

        <Paper sx={{ p: 2 }}>
          <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" alignItems={{ xs: 'flex-start', sm: 'center' }} gap={1}>
            <Box>
              <Typography variant="h6">Session</Typography>
              <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                {auth?.share_auth === false ? 'Local admin mode' : auth?.authenticated ? `Signed in as ${auth.username} (${auth.role})` : 'Not signed in'}
              </Typography>
            </Box>
            {auth?.share_auth !== false && auth?.authenticated && (
              <Button size="small" variant="outlined" onClick={() => apiFetch.logout().then(() => { window.location.href = './login' })}>
                Logout
              </Button>
            )}
          </Stack>
        </Paper>

        {/* Model status */}
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" mb={1.5}>Model</Typography>
          {!isAdmin && <Alert severity="info" sx={{ py: 0, mb: 1 }}>Only admins can load or unload models.</Alert>}
          {memoryMessage && <Alert severity={memoryMessage.severity} sx={{ py: 0, mb: 1 }}>{memoryMessage.text}</Alert>}
          {report?.model_status.loading ? (
            <Stack direction="row" spacing={1.5} alignItems="center">
              <CircularProgress size={18} />
              <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                Model loading… (DiT + VAE + text encoder, ~1–2 min)
              </Typography>
            </Stack>
          ) : report?.model_status.loaded ? (
            <Stack spacing={1}>
              <Chip label="Loaded" color="success" size="small" sx={{ alignSelf: 'flex-start' }} />
              <Typography variant="body2" sx={{ fontFamily: 'Roboto Mono', fontSize: 12, wordBreak: 'break-all' }}>
                {report.model_status.checkpoint}
              </Typography>
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                Quantization: {report.model_status.quantization}
              </Typography>
              {report.model_status.memory?.low_vram?.block_swap_active && (
                <Typography variant="caption" sx={{ color: 'info.main' }}>
                  Low-VRAM: streaming {report.model_status.memory.low_vram.blocks_to_swap} DiT blocks from RAM · encoder offloaded
                </Typography>
              )}
              {report.model_status.text_encoder_source && (
                <Typography variant="caption" sx={{ color: 'text.secondary', wordBreak: 'break-all' }}>
                  Text encoder: {report.model_status.text_encoder_source.kind}
                  {report.model_status.text_encoder_source.runtime ? ` · runtime ${report.model_status.text_encoder_source.runtime}` : ''}
                  {' · '}
                  {report.model_status.text_encoder_source.status || report.model_status.text_encoder_source.path}
                </Typography>
              )}
              <Button variant="outlined" color="error" size="small" onClick={unload} disabled={!isAdmin || !!memoryBusy} sx={{ alignSelf: 'flex-start' }}>
                {memoryBusy === 'unload' ? <CircularProgress size={14} color="inherit" /> : 'Unload'}
              </Button>
            </Stack>
          ) : (
            <Stack spacing={1.5}>
              <Chip label="Not loaded" color="default" size="small" sx={{ alignSelf: 'flex-start' }} />
              {report?.model_status.load_error && (
                <Alert severity="warning" sx={{ py: 0 }}>
                  Auto-load failed: {report.model_status.load_error}
                </Alert>
              )}
              {cpPath && !pathTouched && (
                <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                  Auto-detected checkpoint — just click Load Model.
                </Typography>
              )}
              <TextField
                label="Checkpoint path (.safetensors)"
                value={cpPath}
                onChange={e => { setCpPath(e.target.value); setPathTouched(true) }}
                size="small" fullWidth
                disabled={!isAdmin}
                placeholder="models\krea2\diffusion_models\krea2_turbo_fp8_scaled.safetensors"
              />
              <Stack direction="row" spacing={1}>
                {['fp8', 'bf16', 'fp16'].map(q => (
                  <Chip key={q} label={q} size="small" clickable
                    variant={quant === q ? 'filled' : 'outlined'}
                    color={quant === q ? 'primary' : 'default'}
                    onClick={() => isAdmin && setQuant(q)}
                  />
                ))}
              </Stack>
              <TextField
                label="Block swap (low-VRAM)"
                type="number"
                value={blocksToSwap}
                onChange={e => setBlocksToSwap(Math.max(0, Math.min(28, Number(e.target.value) || 0)))}
                size="small"
                disabled={!isAdmin}
                inputProps={{ min: 0, max: 28, step: 1 }}
                helperText="Stream the last N of 28 DiT blocks from RAM. 0 = off. Try fp8 + 8–16 to run RAW on 24GB (slower)."
              />
              {loadError && <Alert severity="error" sx={{ py: 0 }}>{loadError}</Alert>}
              <Button
                variant="contained" size="small" onClick={loadModel}
                disabled={!isAdmin || loadingModel || !cpPath}
                startIcon={loadingModel ? <CircularProgress size={14} color="inherit" /> : undefined}
                sx={{ alignSelf: 'flex-start' }}
              >
                Load Model
              </Button>
            </Stack>
          )}
        </Paper>

        <Paper sx={{ p: 2 }}>
          <Stack spacing={1.5}>
            <Typography variant="h6">Memory Tools</Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              Free transient encoder/cache memory or inspect duplicate Krea servers before loading the model.
            </Typography>
            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1}>
              <Button
                variant="outlined"
                size="small"
                onClick={releaseTransientMemory}
                disabled={!isAdmin || !!memoryBusy}
                startIcon={memoryBusy === 'release' ? <CircularProgress size={14} color="inherit" /> : undefined}
              >
                Free transient memory
              </Button>
              <Button
                variant="outlined"
                size="small"
                onClick={loadMemoryProcesses}
                disabled={!isAdmin || !!memoryBusy}
                startIcon={memoryBusy === 'processes' ? <CircularProgress size={14} color="inherit" /> : undefined}
              >
                Detect Krea servers
              </Button>
            </Stack>
            {kreaProcesses.length > 0 && (
              <Stack spacing={1}>
                {kreaProcesses.map(proc => (
                  <Paper key={proc.pid} variant="outlined" sx={{ p: 1 }}>
                    <Stack spacing={0.75}>
                      <Typography variant="body2" sx={{ fontFamily: 'Roboto Mono', fontSize: 12 }}>
                        PID {proc.pid}{proc.port ? ` · port ${proc.port}` : ''}{proc.used_memory_gb != null ? ` · ${proc.used_memory_gb.toFixed(1)} GB VRAM` : ''}
                      </Typography>
                      <Typography variant="caption" sx={{ color: 'text.disabled', wordBreak: 'break-all' }}>
                        {proc.command_line}
                      </Typography>
                      <Button
                        variant="outlined"
                        color="warning"
                        size="small"
                        onClick={() => stopMemoryProcess(proc.pid)}
                        disabled={!isAdmin || !proc.can_stop || !!memoryBusy}
                        sx={{ alignSelf: 'flex-start' }}
                      >
                        {memoryBusy === `stop-${proc.pid}` ? <CircularProgress size={14} color="inherit" /> : 'Stop this duplicate'}
                      </Button>
                    </Stack>
                  </Paper>
                ))}
              </Stack>
            )}
          </Stack>
        </Paper>

        {/* Krea conditioning assets */}
        <Paper sx={{ p: 2 }}>
          <Stack spacing={1.5}>
            <Stack direction="row" justifyContent="space-between" alignItems="center">
              <Typography variant="h6">Krea Moodboard Conditioning</Typography>
              <Chip
                size="small"
                label={report?.support_models?.filter(m => !m.optional).every(m => m.installed) ? 'Ready' : 'Needs download'}
                color={report?.support_models?.filter(m => !m.optional).every(m => m.installed) ? 'success' : 'warning'}
              />
            </Stack>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              Moodboard reference images use the local Krea/Qwen3-VL encoder to create conditioning tensors.
              This does not call Krea's servers. These assets are large and are normally downloaded during install or first model load.
            </Typography>
            <Stack spacing={1}>
              {(report?.support_models?.filter(model => !model.optional) ?? []).map(model => (
                <Box key={model.id}>
                  <Stack direction="row" justifyContent="space-between" alignItems="center" gap={1}>
                    <Box>
                      <Typography variant="body2">{model.label}</Typography>
                      <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>
                        {model.repo_id} · {model.purpose}
                      </Typography>
                    </Box>
                    <Chip size="small" label={model.installed ? 'Installed' : 'Missing'} color={model.installed ? 'success' : 'warning'} />
                  </Stack>
                </Box>
              ))}
            </Stack>
            {supportMessage && <Alert severity={supportMessage.severity} sx={{ py: 0 }}>{supportMessage.text}</Alert>}
            <Button
              variant="outlined"
              size="small"
              onClick={downloadSupportModels}
              disabled={!isAdmin || downloadingSupport}
              startIcon={downloadingSupport ? <CircularProgress size={14} color="inherit" /> : undefined}
              sx={{ alignSelf: 'flex-start' }}
            >
              {downloadingSupport ? 'Downloading...' : 'Download / Repair Conditioning Assets'}
            </Button>
          </Stack>
        </Paper>

        {/* Precision editing assets */}
        {isAdmin && <Paper sx={{ p: 2 }}>
          <Stack spacing={1.5}>
            <Stack direction="row" justifyContent="space-between" alignItems="center">
              <Typography variant="h6">Precision Editing / FLUX Fill</Typography>
              <Chip
                size="small"
                label={qualityAssets?.items.find(asset => asset.id === 'flux_fill')?.installed ? 'Ready' : 'Setup needed'}
                color={qualityAssets?.items.find(asset => asset.id === 'flux_fill')?.installed ? 'success' : 'warning'}
              />
            </Stack>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              Strict inpaint and preserve-source outpaint use FLUX Fill when installed. This model is gated on Hugging Face:
              open the model page, accept access, paste a Hugging Face token here, then download the asset.
            </Typography>
            <Alert severity="info" sx={{ py: 0 }}>
              Browser login is not enough for local Python downloads. The server needs a Hugging Face access token with permission for black-forest-labs/FLUX.1-Fill-dev.
            </Alert>
            <Stack spacing={1}>
              {(qualityAssets?.items.filter(asset => asset.id === 'flux_fill') ?? []).map(asset => (
                <Box key={asset.id}>
                  <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" alignItems={{ xs: 'stretch', sm: 'center' }} gap={1}>
                    <Box>
                      <Typography variant="body2">{asset.purpose}</Typography>
                      <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', wordBreak: 'break-all' }}>
                        {asset.repo_id} · {asset.local_path}
                      </Typography>
                    </Box>
                    <Chip size="small" label={asset.installed ? 'Installed' : asset.needs_token ? 'Needs HF token' : 'Missing'} color={asset.installed ? 'success' : 'warning'} />
                  </Stack>
                  {!asset.installed && (
                    <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} sx={{ mt: 1 }}>
                      <Button size="small" variant="outlined" href={asset.setup_url} target="_blank" rel="noreferrer">
                        Open HF model page
                      </Button>
                      <Button
                        size="small"
                        variant="contained"
                        onClick={() => downloadQualityAsset(asset.id)}
                        disabled={qualityBusy === asset.id || asset.needs_token}
                        startIcon={qualityBusy === asset.id ? <CircularProgress size={14} color="inherit" /> : undefined}
                      >
                        {qualityBusy === asset.id ? 'Downloading...' : 'Download FLUX Fill'}
                      </Button>
                    </Stack>
                  )}
                </Box>
              ))}
            </Stack>
            <TextField
              label="Hugging Face access token"
              value={settingsDraft.hf_token}
              onChange={e => setSettingsDraft(d => ({ ...d, hf_token: e.target.value }))}
              size="small"
              fullWidth
              type="password"
              placeholder={settings?.has_hf_token || qualityAssets?.has_hf_token ? 'Token saved for this server session. Paste a new token to replace it.' : 'hf_...'}
              helperText="Use a token from Hugging Face Settings > Access Tokens after accepting the FLUX Fill model access terms."
            />
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              <Button
                variant="outlined"
                size="small"
                onClick={saveHfToken}
                disabled={savingSettings || !settingsDraft.hf_token.trim()}
                startIcon={savingSettings ? <CircularProgress size={14} color="inherit" /> : undefined}
              >
                Save HF Token
              </Button>
              <Button variant="text" size="small" onClick={loadQualityAssets}>
                Refresh status
              </Button>
            </Stack>
            {qualityMessage && <Alert severity={qualityMessage.severity} sx={{ py: 0 }}>{qualityMessage.text}</Alert>}
          </Stack>
        </Paper>}

        {/* Magic wand settings */}
        {isAdmin && <Paper sx={{ p: 2 }}>
          <Stack spacing={1.5}>
            <Stack direction="row" justifyContent="space-between" alignItems="center">
              <Typography variant="h6">Magic Wand</Typography>
              <Stack direction="row" spacing={0.75}>
                {settings?.has_ideogram_api_key && <Chip label="Ideogram key saved" color="success" size="small" />}
                {settings?.has_openrouter_api_key && <Chip label="OpenRouter key saved" color="success" size="small" />}
              </Stack>
            </Stack>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              Local Qwen3-VL is the self-contained helper for prompt expansion and image-to-prompt. OpenRouter and Ideogram remain optional hosted helpers.
            </Typography>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              {(['local', 'openrouter', 'ideogram-json'] as const).map(b => (
                <Chip
                  key={b}
                  label={b === 'local' ? 'Local Qwen3-VL' : b === 'openrouter' ? 'OpenRouter' : 'Ideogram JSON'}
                  clickable
                  variant={settingsDraft.prompt_expander_backend === b ? 'filled' : 'outlined'}
                  color={settingsDraft.prompt_expander_backend === b ? 'primary' : 'default'}
                  onClick={() => setSettingsDraft(d => ({ ...d, prompt_expander_backend: b }))}
                />
              ))}
            </Stack>
            <TextField
              label="Ideogram API key"
              value={settingsDraft.ideogram_api_key}
              onChange={e => setSettingsDraft(d => ({ ...d, ideogram_api_key: e.target.value }))}
              size="small"
              fullWidth
              type="password"
              placeholder={settings?.has_ideogram_api_key ? 'Saved. Paste a new key to replace it.' : 'Ideogram API key'}
              helperText="Used only when Magic Wand backend is Ideogram JSON."
            />
            <TextField
              label="OpenRouter API key"
              value={settingsDraft.openrouter_api_key}
              onChange={e => setSettingsDraft(d => ({ ...d, openrouter_api_key: e.target.value }))}
              size="small"
              fullWidth
              type="password"
              placeholder={settings?.has_openrouter_api_key ? 'Saved. Paste a new key to replace it.' : 'sk-or-v1-...'}
              helperText="Applied to this server session. For persistent keys, set them manually in your local .env."
            />
            <TextField
              label="OpenRouter model"
              value={settingsDraft.openrouter_model}
              onChange={e => setSettingsDraft(d => ({ ...d, openrouter_model: e.target.value }))}
              size="small"
              fullWidth
              placeholder="google/gemma-4-31b-it:free"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={settingsDraft.openrouter_free_only}
                  onChange={e => setSettingsDraft(d => ({ ...d, openrouter_free_only: e.target.checked }))}
                />
              }
              label="Free OpenRouter models only"
            />
            {settingsMessage && <Alert severity={settingsMessage.severity} sx={{ py: 0 }}>{settingsMessage.text}</Alert>}
            <Button
              variant="contained"
              size="small"
              onClick={saveMagicWandSettings}
              disabled={savingSettings}
              startIcon={savingSettings ? <CircularProgress size={14} color="inherit" /> : undefined}
              sx={{ alignSelf: 'flex-start' }}
            >
              Save Magic Wand Settings
            </Button>
          </Stack>
        </Paper>}

        {isAdmin && <Paper sx={{ p: 2 }}>
          <Stack spacing={1.5}>
            <Typography variant="h6">Users</Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              Admins can manage sharing, settings, models, and passwords. Users can generate and use the gallery.
            </Typography>
            <Stack spacing={1}>
              {users.map(user => (
                <Stack key={user.username} direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" alignItems={{ xs: 'stretch', sm: 'center' }} gap={1}>
                  <Typography variant="body2">{user.username}</Typography>
                  <Stack direction="row" spacing={1}>
                    {(['user', 'admin'] as const).map(role => (
                      <Chip
                        key={role}
                        size="small"
                        label={role}
                        clickable
                        variant={user.role === role ? 'filled' : 'outlined'}
                        color={user.role === role ? 'primary' : 'default'}
                        onClick={() => changeUserRole(user.username, role).catch(() => setUserMessage({ severity: 'error', text: 'Could not update role.' }))}
                      />
                    ))}
                    <Button size="small" color="error" onClick={() => removeUser(user.username).catch(() => setUserMessage({ severity: 'error', text: 'Could not remove user.' }))}>
                      Revoke
                    </Button>
                    <Button size="small" onClick={() => resetPassword(user.username)}>
                      Reset
                    </Button>
                  </Stack>
                </Stack>
              ))}
            </Stack>
            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1}>
              <TextField label="Username" size="small" value={newUser.username} onChange={e => setNewUser(u => ({ ...u, username: e.target.value }))} />
              <TextField label="Password" size="small" type="password" value={newUser.password} onChange={e => setNewUser(u => ({ ...u, password: e.target.value }))} />
              <TextField select SelectProps={{ native: true }} label="Role" size="small" value={newUser.role} onChange={e => setNewUser(u => ({ ...u, role: e.target.value as 'admin' | 'user' }))} sx={{ minWidth: 120 }}>
                <option value="user">user</option>
                <option value="admin">admin</option>
              </TextField>
              <Button variant="contained" size="small" onClick={addUser} disabled={!newUser.username || newUser.password.length < 8}>
                Add
              </Button>
            </Stack>
            {userMessage && <Alert severity={userMessage.severity} sx={{ py: 0 }}>{userMessage.text}</Alert>}
          </Stack>
        </Paper>}

        {isAdmin && <Paper sx={{ p: 2 }}>
          <Stack spacing={1.5}>
            <Stack direction="row" justifyContent="space-between" alignItems="center">
              <Typography variant="h6">Tailscale Sharing</Typography>
              <Chip
                size="small"
                label={sharing?.funnel.running ? 'Sharing' : 'Stopped'}
                color={sharing?.funnel.running ? 'success' : 'default'}
              />
            </Stack>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              Public sharing always uses the `/krea` path so other Tailscale funnels can keep their own root URLs.
            </Typography>
            <FormControlLabel
              control={
                <Switch
                  checked={!!settings?.krea_share_auto_funnel}
                  onChange={e => setAutoFunnel(e.target.checked)}
                  disabled={sharingAutoSaving}
                />
              }
              label="Start Tailscale and /krea Funnel automatically when run.bat starts"
            />
            <Typography variant="body2" sx={{ fontFamily: 'Roboto Mono', wordBreak: 'break-all' }}>
              {sharing?.funnel.url || 'No public Krea URL yet.'}
            </Typography>
            {!sharing?.tailscale.installed && (
              <Alert severity="warning" sx={{ py: 0 }}>
                Tailscale is not installed. Install it from {sharing?.tailscale.download_url || 'https://tailscale.com/download/windows'}.
              </Alert>
            )}
            {sharing?.tailscale.installed && !sharing?.tailscale.connected && (
              <Alert severity="warning" sx={{ py: 0 }}>
                Tailscale is installed but may not be logged in. Run `tailscale up`.
              </Alert>
            )}
            {sharingMessage && <Alert severity={sharingMessage.severity} sx={{ py: 0 }}>{sharingMessage.text}</Alert>}
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              <Button size="small" variant="outlined" onClick={loadSharing}>Refresh</Button>
              <Button size="small" variant="outlined" onClick={() => apiFetch.tailscaleUp().then(loadSharing).catch((e: any) => setSharingMessage({ severity: 'error', text: e.message }))}>
                Tailscale Up
              </Button>
              <Button size="small" variant="contained" onClick={startSharing} disabled={sharingBusy || !sharing?.tailscale.installed}>
                Start /krea Funnel
              </Button>
              <Button size="small" color="error" variant="outlined" onClick={stopSharing} disabled={sharingBusy || !sharing?.tailscale.installed}>
                Stop /krea Funnel
              </Button>
            </Stack>
          </Stack>
        </Paper>}

        {report?.attention_acceleration && (
          <Paper sx={{ p: 1.5, border: '1px solid rgba(255,255,255,0.08)' }}>
            <Stack direction="row" justifyContent="space-between" alignItems="center" spacing={1}>
              <Typography variant="body2">Attention acceleration</Typography>
              <Chip
                size="small"
                label={report.attention_acceleration.status.replace(/_/g, ' ')}
                color={report.attention_acceleration.status === 'available_but_off' ? 'info' : 'default'}
              />
            </Stack>
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mt: 0.5 }}>
              {report.attention_acceleration.reason}
            </Typography>
            <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block' }}>
              {report.attention_acceleration.recommendation}
            </Typography>
          </Paper>
        )}

        {/* Variants */}
        {report?.variants.map(v => (
          <Paper key={v.id} sx={{ p: 1.5, border: v.ok ? '1px solid rgba(102,187,106,0.3)' : '1px solid rgba(239,83,80,0.2)' }}>
            <Stack direction="row" justifyContent="space-between" alignItems="center">
              <Typography variant="body2">{v.label}</Typography>
              <Chip size="small" label={v.ok ? 'OK' : 'Blocked'} color={v.ok ? 'success' : 'error'} />
            </Stack>
            {v.blockers.map((b, i) => <Typography key={i} variant="caption" sx={{ color: 'error.light', display: 'block', mt: 0.5 }}>{b}</Typography>)}
            {v.warnings.map((w, i) => <Typography key={i} variant="caption" sx={{ color: 'warning.main', display: 'block' }}>{w}</Typography>)}
          </Paper>
        ))}
      </Stack>
    </Box>
  )
}
