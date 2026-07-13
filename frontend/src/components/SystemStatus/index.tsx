import React, { useEffect, useState } from 'react'
import { Accordion, AccordionDetails, AccordionSummary, Alert, Box, Button, Chip, CircularProgress, Dialog, DialogActions, DialogContent, DialogTitle, FormControlLabel, LinearProgress, Link, MenuItem, Paper, Stack, Switch, TextField, Tooltip, Typography } from '@mui/material'
import GpuIcon from '@mui/icons-material/Memory'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import { apiFetch, publicUrl, type AcceleratorStatus, type AppSettings, type AuthSession, type KreaServerProcess, type ModerationEvent, type ModerationStatus, type QualityAsset, type ShareUser, type SharingStatus, type SystemReport } from '../../api'
import { normalizeKreaDeforumStatus } from '../../lib/kreaDeforumStatus'
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

function SettingsAccordion({
  title,
  summary,
  defaultExpanded = false,
  children,
}: {
  title: string
  summary?: React.ReactNode
  defaultExpanded?: boolean
  children: React.ReactNode
}) {
  return (
    <Accordion
      defaultExpanded={defaultExpanded}
      disableGutters
      sx={{ bgcolor: 'background.paper', borderRadius: 2, '&:before': { display: 'none' } }}
    >
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={1} sx={{ width: '100%', pr: 1 }}>
          <Typography variant="h6">{title}</Typography>
          {summary}
        </Stack>
      </AccordionSummary>
      <AccordionDetails sx={{ pt: 0 }}>
        {children}
      </AccordionDetails>
    </Accordion>
  )
}

function GroupLabel({ children }: { children: React.ReactNode }) {
  return (
    <Typography variant="overline" sx={{ display: 'block', color: 'text.disabled', letterSpacing: 1.5, mt: 1.5, mb: -0.5, fontWeight: 700 }}>
      {children}
    </Typography>
  )
}

export default function SystemStatus() {
  const [report, setReport] = useState<SystemReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [fetchError, setFetchError] = useState('')
  const [cpPath, setCpPath] = useState('')
  const [quant, setQuant] = useState('fp8')
  const [blocksToSwap, setBlocksToSwap] = useState(0)
  const [fp8FastMatmul, setFp8FastMatmul] = useState(false)
  const [torchCompile, setTorchCompile] = useState(false)
  const [vaePath, setVaePath] = useState('')
  const [vaeSaving, setVaeSaving] = useState(false)
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
    civitai_token: '',
    openrouter_api_key: '',
    openrouter_model: 'google/gemma-4-31b-it:free',
    openrouter_free_only: true,
    krea_attention_backend: 'sdpa' as 'sdpa' | 'sage',
    seedvr2_model: '3b' as '3b' | '7b',
    local_llm_backend: 'comfy' as 'comfy' | 'transformers' | 'gguf_server',
    comfy_qwen_model: '2b',
    local_qwen_model_id: '',
    local_qwen_device: 'auto' as 'auto' | 'cuda' | 'cpu',
    gguf_helper_base_url: 'http://127.0.0.1:1234/v1',
    gguf_helper_model: 'BennyDaBall/Krea-2-Engineer-V1-GGUF:Q4_K_M',
    gguf_helper_timeout_sec: 120,
    diffusion_engine: 'native_pytorch' as 'native_pytorch' | 'native_gguf' | 'native_int8_convrot',
    krea2_turbo_int8_path: '',
    krea2_raw_int8_path: '',
    gguf_turbo_path: '',
    gguf_raw_path: '',
    krea2_vae_mode: 'qwen' as 'qwen' | 'comfy_qwen' | 'qwen_wan_blend' | 'wan_experimental',
    krea2_vae_blend_radius: 24,
    krea2_vae_blend_strength: 0.65,
  })
  const [savingSettings, setSavingSettings] = useState(false)
  const [settingsMessage, setSettingsMessage] = useState<{ severity: 'success' | 'warning' | 'error'; text: string } | null>(null)
  const [auth, setAuth] = useState<AuthSession | null>(null)
  const [users, setUsers] = useState<ShareUser[]>([])
  const [sharing, setSharing] = useState<SharingStatus | null>(null)
  const [sharingBusy, setSharingBusy] = useState(false)
  const [sharingMessage, setSharingMessage] = useState<{ severity: 'success' | 'warning' | 'error'; text: string } | null>(null)
  const [sharingAutoSaving, setSharingAutoSaving] = useState(false)
  const [userMessage, setUserMessage] = useState<{ severity: 'success' | 'error'; text: string } | null>(null)
  const [newUser, setNewUser] = useState({ username: '', password: '', role: 'user' as 'admin' | 'user' | 'child' })
  const [resetTarget, setResetTarget] = useState<string | null>(null)
  const [resetPasswordValue, setResetPasswordValue] = useState('')
  const [qualityAssets, setQualityAssets] = useState<{ has_hf_token: boolean; items: QualityAsset[] } | null>(null)
  const [qualityBusy, setQualityBusy] = useState<string | null>(null)
  const [qualityMessage, setQualityMessage] = useState<{ severity: 'success' | 'error'; text: string } | null>(null)
  const [moderationEvents, setModerationEvents] = useState<ModerationEvent[]>([])
  const [moderationStatus, setModerationStatus] = useState<ModerationStatus | null>(null)
  const [moderationBusy, setModerationBusy] = useState(false)
  const [moderationInstallBusy, setModerationInstallBusy] = useState(false)
  const [memoryBusy, setMemoryBusy] = useState<string | null>(null)
  const [memoryMessage, setMemoryMessage] = useState<{ severity: 'success' | 'error' | 'info' | 'warning'; text: string } | null>(null)
  const [kreaProcesses, setKreaProcesses] = useState<KreaServerProcess[]>([])
  const [accelerators, setAccelerators] = useState<AcceleratorStatus | null>(null)
  const [acceleratorBusy, setAcceleratorBusy] = useState<string | null>(null)
  const [acceleratorMessage, setAcceleratorMessage] = useState<{ severity: 'success' | 'error' | 'warning'; text: string } | null>(null)
  const [ggufHelperBusy, setGgufHelperBusy] = useState(false)
  const [ggufRuntimeBusy, setGgufRuntimeBusy] = useState(false)
  const { setSystemReport, setParams } = useStore()
  const isAdmin = auth?.role === 'admin'
  const localQwenChoice = !settingsDraft.local_qwen_model_id
    ? 'default'
    : /Huihui-Qwen3-VL-4B-Instruct-abliterated|qwen3_vl_4b_abliterated/i.test(settingsDraft.local_qwen_model_id)
      ? 'abliterated'
      : 'custom'
  const comfyQwenChoice = /4b|Huihui-Qwen3-VL-4B/i.test(settingsDraft.comfy_qwen_model || '')
    ? '4b'
    : '2b'
  const requiredSupportModels = report?.support_models?.filter(model => !model.optional) ?? []
  const supportReady = requiredSupportModels.length > 0 && requiredSupportModels.every(model => model.installed)
  const supportCachedOnly = !supportReady && requiredSupportModels.some(model => !model.installed && model.legacy_cache_installed)
  const qualityItems = qualityAssets?.items ?? []
  const qualityInstalled = qualityItems.filter(asset => asset.installed).length
  const qualityDownloadableMissing = qualityItems.filter(asset => !asset.installed && asset.download_enabled).length
  const sageInstalled = !!accelerators?.sageattention.installed
  const tritonInstalled = !!accelerators?.triton_windows.installed
  const comfyAccel = accelerators?.comfyui_venv
  const comfyHasAccelerators = !!(comfyAccel?.triton || comfyAccel?.sageattention || comfyAccel?.comfy_kitchen)
  const sageActive = settingsDraft.krea_attention_backend === 'sage'
  const kreaDeforum = normalizeKreaDeforumStatus(settings?.krea_deforum)

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
      setFetchError('Cannot reach the Krea server. Is run.bat still running? In share mode the local port may be dynamic; use the URL printed by run.bat.')
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
      setVaePath(s.krea2_vae_path ?? '')
      setSettingsDraft({
        prompt_expander_backend: s.prompt_expander_backend,
        ideogram_api_key: '',
        hf_token: '',
        civitai_token: '',
        openrouter_api_key: '',
        openrouter_model: s.openrouter_model,
        openrouter_free_only: s.openrouter_free_only,
        krea_attention_backend: s.krea_attention_backend ?? 'sdpa',
        seedvr2_model: (s.seedvr2_model === '7b' ? '7b' : '3b'),
        local_llm_backend: s.local_llm_backend ?? 'comfy',
        comfy_qwen_model: s.comfy_qwen_model ?? '2b',
        local_qwen_model_id: s.local_qwen_model_id ?? '',
        local_qwen_device: s.local_qwen_device ?? 'auto',
        gguf_helper_base_url: s.gguf_helper_base_url ?? 'http://127.0.0.1:1234/v1',
        gguf_helper_model: s.gguf_helper_model ?? 'BennyDaBall/Krea-2-Engineer-V1-GGUF:Q4_K_M',
        gguf_helper_timeout_sec: s.gguf_helper_timeout_sec ?? 120,
        diffusion_engine: s.diffusion_engine ?? 'native_pytorch',
        krea2_turbo_int8_path: s.krea2_turbo_int8_path ?? '',
        krea2_raw_int8_path: s.krea2_raw_int8_path ?? '',
        gguf_turbo_path: s.gguf_turbo_path ?? '',
        gguf_raw_path: s.gguf_raw_path ?? '',
        krea2_vae_mode: s.krea2_vae_mode ?? 'qwen',
        krea2_vae_blend_radius: s.krea2_vae_blend_radius ?? 24,
        krea2_vae_blend_strength: s.krea2_vae_blend_strength ?? 0.65,
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

  const loadAccelerators = async () => {
    try {
      setAccelerators(await apiFetch.acceleratorStatus())
    } catch {
      setAcceleratorMessage({ severity: 'error', text: 'Could not load accelerator status.' })
    }
  }

  const loadModerationEvents = async () => {
    setModerationBusy(true)
    try {
      setModerationStatus(await apiFetch.moderationStatus())
      const data = await apiFetch.moderationEvents('', 100)
      setModerationEvents(data.items)
    } catch {
      setModerationEvents([])
    } finally {
      setModerationBusy(false)
    }
  }

  const installImageClassifier = async () => {
    setModerationInstallBusy(true)
    try {
      await apiFetch.installImageClassifier()
      setModerationStatus(await apiFetch.moderationStatus())
    } catch (e: any) {
      setModerationStatus({ image_classifier_available: false, child_image_moderation: 'install_failed', message: e?.response?.data?.detail ?? e.message ?? 'Image classifier setup failed.' })
    } finally {
      setModerationInstallBusy(false)
    }
  }

  useEffect(() => {
    let presenceTimer: number | undefined
    loadAuth().then(session => {
      loadSettings()
      if (session?.role === 'admin') {
        loadUsers()
        loadSharing()
        loadQualityAssets()
        loadModerationEvents()
        loadAccelerators()
        // Refresh the user list periodically so the online/working presence stays current.
        presenceTimer = window.setInterval(() => { loadUsers().catch(() => {}) }, 20000)
      }
    })
    return () => { if (presenceTimer) window.clearInterval(presenceTimer) }
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
      setQualityMessage({ severity: 'success', text: 'Hugging Face token saved to .env — it now persists across restarts and speeds up model downloads.' })
    } catch (e: any) {
      setQualityMessage({ severity: 'error', text: e?.response?.data?.detail ?? e.message ?? 'Could not save Hugging Face token.' })
    } finally {
      setSavingSettings(false)
    }
  }

  const [savingCivitai, setSavingCivitai] = useState(false)
  const [civitaiMessage, setCivitaiMessage] = useState<{ severity: 'success' | 'error'; text: string } | null>(null)
  const saveCivitaiToken = async () => {
    setSavingCivitai(true)
    setCivitaiMessage(null)
    try {
      await apiFetch.updateSettings({ civitai_token: settingsDraft.civitai_token.trim() })
      setSettingsDraft(d => ({ ...d, civitai_token: '' }))
      await loadSettings()
      setCivitaiMessage({
        severity: 'success',
        text: 'Civitai API key saved to .env — it now persists across restarts.',
      })
    } catch (e: any) {
      setCivitaiMessage({ severity: 'error', text: e?.response?.data?.detail ?? e.message ?? 'Could not save Civitai API key.' })
    } finally {
      setSavingCivitai(false)
    }
  }

  const downloadQualityAsset = async (assetId: string) => {
    setQualityBusy(assetId)
    setQualityMessage(null)
    try {
      await apiFetch.downloadQualityAsset(assetId)
      await loadQualityAssets()
      await refresh()
      setQualityMessage({ severity: 'success', text: 'Asset is ready.' })
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
        local_llm_backend: settingsDraft.local_llm_backend,
        comfy_qwen_model: settingsDraft.comfy_qwen_model,
        local_qwen_model_id: settingsDraft.local_qwen_model_id,
        local_qwen_device: settingsDraft.local_qwen_device,
        gguf_helper_base_url: settingsDraft.gguf_helper_base_url,
        gguf_helper_model: settingsDraft.gguf_helper_model,
        gguf_helper_timeout_sec: settingsDraft.gguf_helper_timeout_sec,
        diffusion_engine: settingsDraft.diffusion_engine,
        gguf_turbo_path: settingsDraft.gguf_turbo_path,
        gguf_raw_path: settingsDraft.gguf_raw_path,
        krea2_vae_mode: settingsDraft.krea2_vae_mode,
        krea2_vae_blend_radius: settingsDraft.krea2_vae_blend_radius,
        krea2_vae_blend_strength: settingsDraft.krea2_vae_blend_strength,
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

  const testGgufHelper = async () => {
    setGgufHelperBusy(true)
    setSettingsMessage(null)
    try {
      await apiFetch.updateSettings({
        local_llm_backend: settingsDraft.local_llm_backend,
        local_qwen_model_id: settingsDraft.local_qwen_model_id,
        local_qwen_device: settingsDraft.local_qwen_device,
        gguf_helper_base_url: settingsDraft.gguf_helper_base_url,
        gguf_helper_model: settingsDraft.gguf_helper_model,
        gguf_helper_timeout_sec: settingsDraft.gguf_helper_timeout_sec,
      })
      const result = await apiFetch.testGgufHelper()
      setSettingsMessage({ severity: 'success', text: `GGUF helper connected: ${result.expanded.slice(0, 140)}` })
      await loadSettings()
    } catch (e: any) {
      setSettingsMessage({ severity: 'error', text: e?.response?.data?.detail ?? e.message ?? 'GGUF helper test failed.' })
    } finally {
      setGgufHelperBusy(false)
    }
  }

  const setupNativeInt8 = async () => {
    setGgufRuntimeBusy(true)
    setSettingsMessage(null)
    try {
      const result = await apiFetch.setupNativeInt8()
      setSettingsDraft(d => ({
        ...d,
        diffusion_engine: result.diffusion_engine,
        krea2_turbo_int8_path: result.turbo_path,
      }))
      setParams({
        diffusion_engine: 'native_int8_convrot',
        model_profile: 'krea_turbo',
        checkpoint: 'turbo',
        quantization: 'int8',
        steps: result.sampler.steps,
        cfg: result.sampler.cfg,
        mu: result.sampler.mu,
        sampler: result.sampler.sampler as any,
        scheduler: result.sampler.scheduler as any,
        resolution_tier: '1k',
        aspect_ratio: '1:1',
        width: 1024,
        height: 1024,
        conditioning_mode: 'auto',
        negative_prompt: '',
      })
      setCpPath(result.turbo_path)
      setQuant(result.quantization)
      await loadQualityAssets()
      await loadSettings()
      setSettingsMessage({
        severity: 'success',
        text: `Native INT8 setup applied. ${result.assets.filter(asset => asset.skipped).length}/${result.assets.length} assets were already installed. ${result.warnings.join(' ')}`,
      })
    } catch (e: any) {
      setSettingsMessage({ severity: 'error', text: e?.response?.data?.detail ?? e.message ?? 'Native INT8 setup failed.' })
    } finally {
      setGgufRuntimeBusy(false)
    }
  }

  const setupGgufLowVram = async () => {
    setGgufRuntimeBusy(true)
    setSettingsMessage(null)
    try {
      const result = await apiFetch.setupGgufLowVram()
      setSettingsDraft(d => ({
        ...d,
        diffusion_engine: result.diffusion_engine,
        gguf_turbo_path: result.turbo_path,
      }))
      setParams({
        diffusion_engine: 'native_gguf',
        model_profile: 'krea_turbo',
        checkpoint: 'turbo',
        quantization: 'gguf',
        steps: result.sampler.steps,
        cfg: result.sampler.cfg,
        mu: result.sampler.mu,
        sampler: result.sampler.sampler as any,
        scheduler: result.sampler.scheduler as any,
        resolution_tier: '1k',
        aspect_ratio: '1:1',
        width: 1024,
        height: 1024,
        num_images: 1,
        cfg_zero_star: false,
        conditioning_mode: 'auto',
        negative_prompt: '',
      })
      setCpPath(result.checkpoint_path)
      setQuant(result.quantization)
      await loadQualityAssets()
      await loadSettings()
      setSettingsMessage({
        severity: 'success',
        text: `GGUF low-VRAM setup applied. ${result.assets.filter(asset => asset.skipped).length}/${result.assets.length} assets were already installed. ${result.warnings.join(' ')}`,
      })
    } catch (e: any) {
      setSettingsMessage({ severity: 'error', text: e?.response?.data?.detail ?? e.message ?? 'GGUF low-VRAM setup failed.' })
    } finally {
      setGgufRuntimeBusy(false)
    }
  }

  const saveAttentionBackend = async (backend: 'sdpa' | 'sage') => {
    setAcceleratorBusy('save')
    setAcceleratorMessage(null)
    try {
      await apiFetch.updateSettings({ krea_attention_backend: backend })
      setSettingsDraft(d => ({ ...d, krea_attention_backend: backend }))
      await loadSettings()
      setAcceleratorMessage({
        severity: backend === 'sage' ? 'warning' : 'success',
        text: backend === 'sage'
          ? 'SageAttention enabled for A/B testing. Verify fixed-seed outputs before keeping it on.'
          : 'SDPA restored as the stable attention backend.',
      })
    } catch (e: any) {
      setAcceleratorMessage({ severity: 'error', text: e?.response?.data?.detail ?? e.message ?? 'Could not save attention backend.' })
    } finally {
      setAcceleratorBusy(null)
    }
  }

  const saveSeedvr2Model = async (model: '3b' | '7b') => {
    setAcceleratorBusy('save')
    setAcceleratorMessage(null)
    try {
      await apiFetch.updateSettings({ seedvr2_model: model })
      setSettingsDraft(d => ({ ...d, seedvr2_model: model }))
      await loadSettings()
      setAcceleratorMessage({
        severity: model === '7b' ? 'warning' : 'success',
        text: model === '7b'
          ? 'SeedVR2 7B fp16 selected (max quality). First 4K upscale streams blocks to RAM — slower but sharper.'
          : 'SeedVR2 3B fp8 restored (fast default).',
      })
    } catch (e: any) {
      setAcceleratorMessage({ severity: 'error', text: e?.response?.data?.detail ?? e.message ?? 'Could not save SeedVR2 model.' })
    } finally {
      setAcceleratorBusy(null)
    }
  }

  const installAccelerator = async (kind: 'triton' | 'sage') => {
    setAcceleratorBusy(kind)
    setAcceleratorMessage(null)
    try {
      const result = kind === 'triton' ? await apiFetch.installTritonWindows() : await apiFetch.installSageAttention()
      setAccelerators(result.status)
      setAcceleratorMessage({ severity: 'success', text: kind === 'triton' ? 'Triton for Windows install completed.' : 'SageAttention install completed.' })
    } catch (e: any) {
      setAcceleratorMessage({ severity: 'error', text: e?.response?.data?.detail ?? e.message ?? 'Accelerator install failed.' })
    } finally {
      setAcceleratorBusy(null)
    }
  }

  const loadModel = async () => {
    if (!cpPath) return
    setLoadingModel(true); setLoadError('')
    try {
      await apiFetch.loadModel(cpPath, quant, blocksToSwap, fp8FastMatmul, torchCompile)
      await refresh()
    } catch (e: any) {
      setLoadError(e?.response?.data?.detail ?? e.message)
    } finally { setLoadingModel(false) }
  }

  const preflightLoadModel = async () => {
    if (!cpPath) return
    setLoadingModel(true); setLoadError('')
    try {
      const result = await apiFetch.preflightLoadModel(cpPath, quant, blocksToSwap, fp8FastMatmul, torchCompile)
      if (result.ok) {
        setLoadError('')
        setMemoryMessage({ severity: 'success', text: result.detail })
      } else {
        setLoadError(result.detail)
        setMemoryMessage({ severity: 'warning', text: result.detail })
      }
      await refresh()
    } catch (e: any) {
      setLoadError(e?.response?.data?.detail ?? e.message ?? 'Model preflight failed.')
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

  const safeCleanMemory = async () => {
    setMemoryBusy('safe-clean')
    setMemoryMessage(null)
    try {
      const result = await apiFetch.safeCleanMemory()
      await refresh()
      const cleared = result.cleared_conditioning_entries ?? 0
      setMemoryMessage({
        severity: 'success',
        text: `Safe RAM clean complete. Helper cache ${result.helper_unloaded ? 'cleared' : 'not loaded'}; conditioning entries cleared: ${cleared}.`,
      })
    } catch (e: any) {
      setMemoryMessage({ severity: 'error', text: e?.response?.data?.detail ?? e.message ?? 'Could not run safe RAM clean.' })
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

  const changeUserRole = async (username: string, role: 'admin' | 'user' | 'child') => {
    setUsers(await apiFetch.setUserRole(username, role))
  }

  const removeUser = async (username: string) => {
    setUsers(await apiFetch.removeUser(username))
  }

  const resetPassword = (username: string) => {
    setResetTarget(username)
    setResetPasswordValue('')
  }

  const runResetPassword = async () => {
    const username = resetTarget
    const password = resetPasswordValue
    setResetTarget(null)
    if (!username || !password) return
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

  const repairSharing = async () => {
    setSharingBusy(true); setSharingMessage(null)
    try {
      const result = await apiFetch.repairSharing()
      await loadSharing()
      setSharingMessage({
        severity: result.ok ? 'success' : result.needs_admin_service_restart ? 'error' : 'warning',
        text: result.message,
      })
    } catch (e: any) {
      setSharingMessage({ severity: 'error', text: e?.response?.data?.detail ?? e.message ?? 'Could not repair sharing.' })
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
        <GroupLabel>System</GroupLabel>
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
            {auth?.share_auth !== false && auth && !auth.authenticated && (
              <Button size="small" variant="contained" onClick={() => { window.location.href = './login' }}>
                Sign in
              </Button>
            )}
          </Stack>
        </Paper>

        <Paper variant="outlined" sx={{ p: 2 }}>
          <Stack spacing={1.25}>
            <Stack
              direction={{ xs: 'column', sm: 'row' }}
              justifyContent="space-between"
              alignItems={{ xs: 'flex-start', sm: 'center' }}
              gap={1}
            >
              <Box>
                <Typography variant="h6">KreaDeforum / Animate</Typography>
                <Typography variant="body2" color="text.secondary">
                  External ComfyUI animation dependency
                </Typography>
              </Box>
              <Chip
                size="small"
                label={kreaDeforum.available ? 'Ready' : 'Setup needed'}
                color={kreaDeforum.available ? 'success' : 'warning'}
              />
            </Stack>
            {settings ? (
              <>
                <Stack direction="row" spacing={0.75} useFlexGap flexWrap="wrap">
                  <Chip size="small" variant="outlined" label={`Revision ${kreaDeforum.revision.slice(0, 10)}`} />
                  <Chip size="small" variant="outlined" label={`Patch ${kreaDeforum.patch_version}`} />
                  <Chip
                    size="small"
                    variant="outlined"
                    color={kreaDeforum.incompatible_capabilities.length ? 'warning' : 'success'}
                    label={kreaDeforum.incompatible_capabilities.length ? 'Patch incompatible' : 'Chunking capability OK'}
                  />
                  <Chip
                    size="small"
                    variant="outlined"
                    color={kreaDeforum.midas_ready ? 'success' : 'warning'}
                    label={`MiDaS / 3D ${kreaDeforum.midas_ready ? 'ready' : 'not ready'}`}
                  />
                </Stack>
                {!!kreaDeforum.missing_nodes.length && (
                  <Alert severity="warning" sx={{ py: 0 }}>
                    Missing nodes: {kreaDeforum.missing_nodes.join(', ')}
                  </Alert>
                )}
                {!!kreaDeforum.incompatible_capabilities.length && (
                  <Alert severity="warning" sx={{ py: 0 }}>
                    {kreaDeforum.incompatible_capabilities.join(', ')}
                  </Alert>
                )}
                {!kreaDeforum.available && (
                  <Typography variant="body2" color="text.secondary">
                    Run install.bat, then restart ComfyUI. There is no in-app setup action for this external dependency.
                  </Typography>
                )}
                {!kreaDeforum.midas_ready && (
                  <Typography variant="body2" color="text.secondary">
                    3D only: {kreaDeforum.midas_reason}
                  </Typography>
                )}
                <Typography variant="caption" color="text.disabled">
                  External component · license {kreaDeforum.license}
                  {kreaDeforum.probe_failed ? ' · readiness probe failed' : ''}
                  {kreaDeforum.stale ? ' · showing last known status' : ''}
                </Typography>
              </>
            ) : (
              <Typography variant="body2" color="text.secondary">Checking animation dependencies…</Typography>
            )}
          </Stack>
        </Paper>

        {isAdmin && <SettingsAccordion
          title="Experimental Accelerators"
          summary={
            <Stack direction="row" spacing={0.75}>
              <Chip size="small" color={comfyHasAccelerators ? 'success' : 'default'} label={comfyHasAccelerators ? 'ComfyUI: Triton+Sage' : 'ComfyUI accel missing'} />
              <Chip size="small" color="info" label="Image gens use ComfyUI" />
            </Stack>
          }
        >
          <Stack spacing={1.25}>
            <Stack direction="row" justifyContent="space-between" alignItems="center">
              <Button size="small" variant="text" onClick={loadAccelerators}>Refresh</Button>
            </Stack>
            <Alert severity={comfyHasAccelerators ? 'success' : 'warning'} sx={{ py: 0 }}>
              {comfyHasAccelerators
                ? 'ComfyUI has Triton + SageAttention and uses them automatically for image generation (including INT8 OTU). No Studio toggle needed.'
                : 'ComfyUI is missing Triton/SageAttention wheels — re-run the ComfyUI installer scripts to restore acceleration.'}
            </Alert>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              The Studio Python venv no longer runs the image model (native DiT is deprecated). Install buttons below only affect the Studio helper venv (moodboard Qwen / magic wand), not ComfyUI.
            </Typography>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              <Chip size="small" color={comfyAccel?.triton ? 'success' : 'default'} label={`Comfy Triton ${comfyAccel?.triton ? 'ok' : 'missing'}`} />
              <Chip size="small" color={comfyAccel?.sageattention ? 'success' : 'default'} label={`Comfy Sage ${comfyAccel?.sageattention ? 'ok' : 'missing'}`} />
              <Chip size="small" color={tritonInstalled ? 'success' : 'default'} label={`Studio Triton ${tritonInstalled ? 'ok' : 'n/a'}`} />
              <Chip size="small" color={sageInstalled ? 'success' : 'default'} label={`Studio Sage ${sageInstalled ? 'ok' : 'n/a'}`} />
            </Stack>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              <Button
                size="small"
                variant="outlined"
                disabled={!!acceleratorBusy || tritonInstalled}
                startIcon={acceleratorBusy === 'triton' ? <CircularProgress size={14} /> : undefined}
                onClick={() => installAccelerator('triton')}
              >
                {tritonInstalled ? 'Studio Triton Present' : 'Install Triton into Studio venv'}
              </Button>
              <Button
                size="small"
                variant="outlined"
                disabled={!!acceleratorBusy || sageInstalled}
                startIcon={acceleratorBusy === 'sage' ? <CircularProgress size={14} /> : undefined}
                onClick={() => installAccelerator('sage')}
              >
                {sageInstalled ? 'Studio Sage Present' : 'Install Sage into Studio venv'}
              </Button>
            </Stack>
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={settingsDraft.seedvr2_model === '7b'}
                  disabled={!!acceleratorBusy}
                  onChange={e => saveSeedvr2Model(e.target.checked ? '7b' : '3b')}
                />
              }
              label={
                <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                  SeedVR2 upscaler: {settingsDraft.seedvr2_model === '7b' ? '7B fp16 (max quality)' : '3B fp8 (fast default)'} — used for RAW-4K auto-upscale
                </Typography>
              }
            />
            {acceleratorMessage && <Alert severity={acceleratorMessage.severity} sx={{ py: 0 }}>{acceleratorMessage.text}</Alert>}
          </Stack>
        </SettingsAccordion>}

        {/* GPU profile + per-system recommendation */}
        {report?.gpu_capabilities && (
          <SettingsAccordion title="GPU Profile">
            <Stack spacing={0.5}>
              {report.runnability && (
                <Alert severity={report.runnability.can_run ? (report.runnability.tier === 'minimum' ? 'warning' : 'success') : 'error'} sx={{ py: 0, mb: 0.5 }}>
                  {report.runnability.can_run
                    ? `Can run — ${report.runnability.tier} tier (${report.runnability.compute_dtype} compute). ${report.runnability.reason}`
                    : `Cannot run: ${report.runnability.reason}`}
                </Alert>
              )}
              <Typography variant="body2" sx={{ wordBreak: 'break-all' }}>
                {report.gpu_capabilities.name || 'GPU'} · {report.gpu_capabilities.arch}
                {report.gpu_capabilities.compute_capability ? ` (sm_${report.gpu_capabilities.compute_capability.replace('.', '')})` : ''}
                {report.gpu_capabilities.vram_total_gb != null ? ` · ${report.gpu_capabilities.vram_total_gb}GB` : ''}
              </Typography>
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                bf16 {report.gpu_capabilities.supports_bf16 ? '✓' : '✗'} ·
                {' '}fp8 compute {report.gpu_capabilities.supports_fp8_compute ? '✓' : '✗'} ·
                {' '}nvfp4 {report.gpu_capabilities.supports_nvfp4 ? '✓' : '✗'}
              </Typography>
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>{report.gpu_capabilities.fp8_note}</Typography>
              {report.recommended_runtime && (
                <Alert severity="info" sx={{ py: 0, mt: 0.5 }}>
                  Recommended: <b>{report.recommended_runtime.quantization}</b>
                  {report.recommended_runtime.blocks_to_swap ? `, block-swap ~${report.recommended_runtime.blocks_to_swap}` : ', no block-swap'}
                  , up to <b>{report.recommended_runtime.max_tier.toUpperCase()}</b>.
                  {report.recommended_runtime.notes ? ` ${report.recommended_runtime.notes}` : ''}
                </Alert>
              )}
            </Stack>
          </SettingsAccordion>
        )}

        <GroupLabel>Models &amp; Generation</GroupLabel>
        {/* Model status */}
        <SettingsAccordion title="Model" defaultExpanded>
          {memoryMessage && <Alert severity={memoryMessage.severity} sx={{ py: 0, mb: 1 }}>{memoryMessage.text}</Alert>}
          {report?.model_status.backend === 'comfyui' ? (
            /* ComfyUI is the generation engine: no in-process model to load or
               unload. Show engine health instead of the legacy native controls. */
            <Stack spacing={1}>
              <Stack direction="row" spacing={1} alignItems="center">
                <Chip
                  label={report.model_status.loaded ? 'ComfyUI engine online' : 'ComfyUI engine unreachable'}
                  color={report.model_status.loaded ? 'success' : 'error'}
                  size="small"
                />
                <Chip label="models load on demand" size="small" variant="outlined" />
              </Stack>
              <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                All image generation runs in ComfyUI, which loads the right checkpoint per job — there is nothing to load or unload here.
              </Typography>
              {report.model_status.auto_checkpoint && (
                <Typography variant="caption" sx={{ color: 'text.disabled', fontFamily: 'Roboto Mono', fontSize: 11, wordBreak: 'break-all' }}>
                  Default checkpoint: {report.model_status.auto_checkpoint} ({report.model_status.auto_quant})
                </Typography>
              )}
              {!report.model_status.loaded && (
                <Alert severity="warning" sx={{ py: 0 }}>
                  ComfyUI is not responding. It normally starts with run.bat — give it a minute, or restart the app.
                  {isAdmin ? ' Check logs\\comfyui.err.log if it stays offline.' : ''}
                </Alert>
              )}
            </Stack>
          ) : report?.model_status.loading ? (
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
                {['fp8', 'gguf', 'bf16', 'fp16'].map(q => (
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
              <FormControlLabel
                control={
                  <Switch
                    size="small"
                    checked={fp8FastMatmul}
                    onChange={e => setFp8FastMatmul(e.target.checked)}
                    disabled={!isAdmin || quant !== 'fp8' || !report?.gpu_capabilities?.supports_fp8_compute}
                  />
                }
                label={
                  <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                    fp8 fast matmul (experimental){report && !report.gpu_capabilities?.supports_fp8_compute ? ' — needs Ada/Blackwell' : quant !== 'fp8' ? ' — fp8 only' : ' — faster on Ada/Blackwell'}
                  </Typography>
                }
              />
              <FormControlLabel
                control={
                  <Switch
                    size="small"
                    checked={torchCompile}
                    onChange={e => setTorchCompile(e.target.checked)}
                    disabled={!isAdmin || blocksToSwap > 0}
                  />
                }
                label={
                  <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                    torch.compile (experimental){blocksToSwap > 0 ? ' — disable block swap to use' : ' — needs Triton/inductor; first gen slower'}
                  </Typography>
                }
              />
              <Stack direction="row" spacing={1} alignItems="flex-start">
                <TextField
                  select
                  label="VAE decoder mode"
                  value={settingsDraft.krea2_vae_mode}
                  onChange={e => setSettingsDraft(d => ({ ...d, krea2_vae_mode: e.target.value as typeof d.krea2_vae_mode }))}
                  size="small"
                  disabled={!isAdmin}
                  sx={{ minWidth: 260 }}
                  helperText="Default is stock Qwen. Wan modes are optional experiments."
                >
                  <MenuItem value="qwen">Qwen VAE (default)</MenuItem>
                  <MenuItem value="comfy_qwen">Comfy Qwen VAE</MenuItem>
                  <MenuItem value="qwen_wan_blend">Qwen + Wan detail blend</MenuItem>
                  <MenuItem value="wan_experimental">Generic Wan 2.1 (experimental)</MenuItem>
                </TextField>
                <TextField
                  label="VAE path"
                  value={vaePath}
                  onChange={e => setVaePath(e.target.value)}
                  size="small"
                  fullWidth
                  disabled={!isAdmin}
                  placeholder="Optional manual VAE override path"
                  helperText="Optional manual override. Leave empty for selected mode's default asset. Applies on next model load."
                />
                <Button
                  variant="outlined" size="small" sx={{ mt: 0.5 }}
                  disabled={!isAdmin || vaeSaving}
                  onClick={async () => {
                    setVaeSaving(true)
                    try {
                      await apiFetch.updateSettings({
                        krea2_vae_path: vaePath,
                        krea2_vae_mode: settingsDraft.krea2_vae_mode,
                        krea2_vae_blend_radius: settingsDraft.krea2_vae_blend_radius,
                        krea2_vae_blend_strength: settingsDraft.krea2_vae_blend_strength,
                      })
                      setSettingsMessage({ severity: 'success', text: 'VAE settings saved. Reload the model to apply.' })
                    } catch (e: any) {
                      setSettingsMessage({ severity: 'error', text: e?.response?.data?.detail ?? 'Could not save VAE path.' })
                    } finally {
                      setVaeSaving(false)
                    }
                  }}
                >
                  Save
                </Button>
              </Stack>
              {settingsDraft.krea2_vae_mode === 'qwen_wan_blend' && (
                <Stack direction="row" spacing={1}>
                  <TextField
                    label="Blend blur radius"
                    type="number"
                    size="small"
                    value={settingsDraft.krea2_vae_blend_radius}
                    onChange={e => setSettingsDraft(d => ({ ...d, krea2_vae_blend_radius: Math.max(1, Number(e.target.value) || 24) }))}
                    inputProps={{ min: 1, max: 128, step: 1 }}
                    helperText="Start 15-30"
                  />
                  <TextField
                    label="Wan detail strength"
                    type="number"
                    size="small"
                    value={settingsDraft.krea2_vae_blend_strength}
                    onChange={e => setSettingsDraft(d => ({ ...d, krea2_vae_blend_strength: Math.max(0, Math.min(2, Number(e.target.value) || 0.65)) }))}
                    inputProps={{ min: 0, max: 2, step: 0.05 }}
                    helperText="Start 0.5-0.8"
                  />
                </Stack>
              )}
              {loadError && <Alert severity="error" sx={{ py: 0 }}>{loadError}</Alert>}
              <Stack direction="row" spacing={1} flexWrap="wrap">
                <Button
                  variant="outlined" size="small" onClick={preflightLoadModel}
                  disabled={!isAdmin || loadingModel || !cpPath}
                  startIcon={loadingModel ? <CircularProgress size={14} color="inherit" /> : undefined}
                >
                  Can I load this?
                </Button>
                <Button
                  variant="contained" size="small" onClick={loadModel}
                  disabled={!isAdmin || loadingModel || !cpPath}
                  startIcon={loadingModel ? <CircularProgress size={14} color="inherit" /> : undefined}
                >
                  Load Model
                </Button>
              </Stack>
            </Stack>
          )}
        </SettingsAccordion>

        <SettingsAccordion title="Memory Tools">
          <Stack spacing={1.5}>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              Free transient encoder/cache memory or inspect duplicate Krea servers before loading the model.
            </Typography>
            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1}>
              <Button
                variant="outlined"
                size="small"
                onClick={safeCleanMemory}
                disabled={!isAdmin || !!memoryBusy}
                startIcon={memoryBusy === 'safe-clean' ? <CircularProgress size={14} color="inherit" /> : undefined}
              >
                Safe RAM clean
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
        </SettingsAccordion>

        {/* Krea conditioning assets */}
        <SettingsAccordion
          title="Krea Moodboard Conditioning"
          defaultExpanded={!supportReady}
          summary={
            <Chip
              size="small"
              label={supportReady ? 'Ready' : supportCachedOnly ? 'Cached, repair local copy' : 'Needs download'}
              color={supportReady ? 'success' : supportCachedOnly ? 'info' : 'warning'}
            />
          }
        >
          <Stack spacing={1.5}>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              Moodboard reference images use the local Krea/Qwen3-VL encoder to create conditioning tensors.
              This does not call Krea's servers. If an item is cached but not local, the repair button copies it into the app's expected folder.
            </Typography>
            <Stack spacing={1}>
              {requiredSupportModels.map(model => (
                <Box key={model.id}>
                  <Stack direction="row" justifyContent="space-between" alignItems="center" gap={1}>
                    <Box>
                      <Typography variant="body2">{model.label}</Typography>
                      <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>
                        {model.repo_id} · {model.purpose}
                      </Typography>
                      <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', wordBreak: 'break-all' }}>
                        Local: {model.path || model.cache_dir}
                      </Typography>
                    </Box>
                    <Chip
                      size="small"
                      label={model.installed ? 'Local' : model.legacy_cache_installed ? 'Cached' : 'Missing'}
                      color={model.installed ? 'success' : model.legacy_cache_installed ? 'info' : 'warning'}
                    />
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
        </SettingsAccordion>

        <GroupLabel>API Keys &amp; Helpers</GroupLabel>
        {/* Hugging Face access token (higher download rate limits + gated models) */}
        {isAdmin && <SettingsAccordion
          title="Hugging Face access token"
          defaultExpanded={!(settings?.has_hf_token || qualityAssets?.has_hf_token)}
          summary={(settings?.has_hf_token || qualityAssets?.has_hf_token) ? <Chip label="Token saved" color="success" size="small" /> : <Chip label="Not set" size="small" variant="outlined" />}
        >
          <Stack spacing={1.5}>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              Optional. A Hugging Face token gives the server higher download rate limits and access to gated models you've been granted. Keys saved here are written to .env, so they persist across restarts and speed up model downloads.
            </Typography>
            <TextField
              label="Hugging Face access token"
              value={settingsDraft.hf_token}
              onChange={e => setSettingsDraft(d => ({ ...d, hf_token: e.target.value }))}
              size="small"
              fullWidth
              type="password"
              placeholder={settings?.has_hf_token || qualityAssets?.has_hf_token ? 'Token available. Paste a new token to replace it.' : 'hf_...'}
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
        </SettingsAccordion>}

        {/* Civitai API key */}
        {isAdmin && <SettingsAccordion
          title="Civitai API key"
          defaultExpanded={!settings?.has_civitai_token}
          summary={settings?.has_civitai_token ? <Chip label="Key saved" color="success" size="small" /> : <Chip label="Not set" size="small" variant="outlined" />}
        >
          <Stack spacing={1.5}>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              A free Civitai API key lets the LoRA browser download login-gated models. Keys saved here are written to .env, so they persist across restarts. Create one at{' '}
              <Link href="https://civitai.com/user/account" target="_blank" rel="noreferrer">civitai.com → Account → API Keys</Link>. It's free — Civitai downloads don't cost anything.
            </Typography>
            <TextField
              label="Civitai API key"
              value={settingsDraft.civitai_token}
              onChange={e => setSettingsDraft(d => ({ ...d, civitai_token: e.target.value }))}
              size="small"
              fullWidth
              type="password"
              placeholder={settings?.has_civitai_token ? 'Key available. Paste a new key to replace it.' : 'Paste your Civitai API key...'}
            />
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              <Button
                variant="outlined"
                size="small"
                onClick={saveCivitaiToken}
                disabled={savingCivitai || !settingsDraft.civitai_token.trim()}
                startIcon={savingCivitai ? <CircularProgress size={14} color="inherit" /> : undefined}
              >
                Save Civitai Key
              </Button>
            </Stack>
            {civitaiMessage && <Alert severity={civitaiMessage.severity} sx={{ py: 0 }}>{civitaiMessage.text}</Alert>}
          </Stack>
        </SettingsAccordion>}

        {/* Magic wand settings */}
        {isAdmin && <SettingsAccordion
          title="Magic Wand"
          summary={
            <Stack direction="row" spacing={0.75}>
              {settings?.has_ideogram_api_key && <Chip label="Ideogram key saved" color="success" size="small" />}
              {settings?.has_openrouter_api_key && <Chip label="OpenRouter key saved" color="success" size="small" />}
            </Stack>
          }
        >
          <Stack spacing={1.5}>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              ComfyUI QwenVL is the default helper for prompt expansion, image-to-prompt, and moodboard enrich. Default model is Huihui 2B abliterated (faster); switch to 4B for richer prose. Transformers remains a Studio fallback. GGUF helper is text-only. OpenRouter and Ideogram remain optional hosted helpers.
            </Typography>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              {(['comfy', 'transformers', 'gguf_server'] as const).map(b => (
                <Chip
                  key={b}
                  label={b === 'comfy' ? 'ComfyUI QwenVL' : b === 'transformers' ? 'Transformers Qwen3-VL' : 'GGUF server'}
                  clickable
                  variant={settingsDraft.local_llm_backend === b ? 'filled' : 'outlined'}
                  color={settingsDraft.local_llm_backend === b ? 'secondary' : 'default'}
                  onClick={() => setSettingsDraft(d => ({ ...d, local_llm_backend: b }))}
                />
              ))}
            </Stack>
            {settingsDraft.local_llm_backend === 'comfy' && (
              <Stack spacing={0.75}>
                <Typography variant="caption" sx={{ color: 'text.secondary' }}>Comfy helper model</Typography>
                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                  {([
                    { id: '2b', label: '2B abliterated (default)' },
                    { id: '4b', label: '4B abliterated' },
                  ] as const).map(opt => (
                    <Chip
                      key={opt.id}
                      label={opt.label}
                      clickable
                      variant={comfyQwenChoice === opt.id ? 'filled' : 'outlined'}
                      color={comfyQwenChoice === opt.id ? 'secondary' : 'default'}
                      onClick={() => setSettingsDraft(d => ({ ...d, comfy_qwen_model: opt.id }))}
                    />
                  ))}
                </Stack>
              </Stack>
            )}
            {settingsDraft.local_llm_backend === 'transformers' && (
              <Stack spacing={1}>
                <TextField
                  select
                  label="Local Qwen model"
                  value={localQwenChoice}
                  onChange={e => {
                    const choice = e.target.value
                    setSettingsDraft(d => ({
                      ...d,
                      local_qwen_model_id: choice === 'default'
                        ? ''
                        : choice === 'abliterated'
                          ? 'huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated'
                          : d.local_qwen_model_id || 'custom/repo-or-path',
                    }))
                  }}
                  size="small"
                  fullWidth
                  helperText="Transformers fallback only. Comfy FP8 abliterated CLIP is separate (generation path)."
                >
                  <MenuItem value="default">Default installed Qwen3-VL</MenuItem>
                  <MenuItem value="abliterated">Abliterated Qwen3-VL 4B</MenuItem>
                  <MenuItem value="custom">Custom repo/path</MenuItem>
                </TextField>
                {localQwenChoice === 'custom' && (
                  <TextField
                    label="Custom Qwen repo/path"
                    value={settingsDraft.local_qwen_model_id}
                    onChange={e => setSettingsDraft(d => ({ ...d, local_qwen_model_id: e.target.value }))}
                    size="small"
                    fullWidth
                    placeholder="HF repo id or local model folder"
                  />
                )}
                <TextField
                  select
                  label="Local Qwen device"
                  value={settingsDraft.local_qwen_device}
                  onChange={e => setSettingsDraft(d => ({ ...d, local_qwen_device: e.target.value as typeof d.local_qwen_device }))}
                  size="small"
                  fullWidth
                  helperText="Auto uses CUDA only when safe; otherwise warns fast. CPU is explicit because it can be very slow."
                >
                  <MenuItem value="auto">Auto safe / fast fail</MenuItem>
                  <MenuItem value="cpu">CPU slow</MenuItem>
                  <MenuItem value="cuda">CUDA</MenuItem>
                </TextField>
              </Stack>
            )}
            {settingsDraft.local_llm_backend === 'gguf_server' && (
              <Stack spacing={1}>
                <Alert severity="info" sx={{ py: 0 }}>
                  GGUF helper is text-only. Prompt expansion and planner can use it; image description and image-based moodboard authoring still use Qwen3-VL/OpenRouter.
                </Alert>
                <TextField
                  label="GGUF helper base URL"
                  value={settingsDraft.gguf_helper_base_url}
                  onChange={e => setSettingsDraft(d => ({ ...d, gguf_helper_base_url: e.target.value }))}
                  size="small"
                  fullWidth
                  placeholder="http://127.0.0.1:1234/v1"
                  helperText="OpenAI-compatible local endpoint from LM Studio, llama-server, or similar."
                />
                <TextField
                  label="GGUF helper model"
                  value={settingsDraft.gguf_helper_model}
                  onChange={e => setSettingsDraft(d => ({ ...d, gguf_helper_model: e.target.value }))}
                  size="small"
                  fullWidth
                  placeholder="BennyDaBall/Krea-2-Engineer-V1-GGUF:Q4_K_M"
                />
                <TextField
                  label="GGUF helper timeout (seconds)"
                  type="number"
                  value={settingsDraft.gguf_helper_timeout_sec}
                  onChange={e => setSettingsDraft(d => ({ ...d, gguf_helper_timeout_sec: Math.max(10, Number(e.target.value) || 120) }))}
                  size="small"
                  inputProps={{ min: 10, step: 10 }}
                />
                <Button
                  variant="outlined"
                  size="small"
                  onClick={testGgufHelper}
                  disabled={ggufHelperBusy}
                  startIcon={ggufHelperBusy ? <CircularProgress size={14} color="inherit" /> : undefined}
                  sx={{ alignSelf: 'flex-start' }}
                >
                  Test GGUF Helper
                </Button>
              </Stack>
            )}
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
              placeholder={settings?.has_ideogram_api_key ? 'Key available. Paste a new key to replace it for this session.' : 'Ideogram API key'}
              helperText="Used only when Magic Wand backend is Ideogram JSON. For persistence, set IDEOGRAM_API_KEY in .env."
            />
            <TextField
              label="OpenRouter API key"
              value={settingsDraft.openrouter_api_key}
              onChange={e => setSettingsDraft(d => ({ ...d, openrouter_api_key: e.target.value }))}
              size="small"
              fullWidth
              type="password"
              placeholder={settings?.has_openrouter_api_key ? 'Key available. Paste a new key to replace it for this session.' : 'sk-or-v1-...'}
              helperText="For persistence, set OPENROUTER_API_KEY in .env. Saving other settings will not remove it."
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
        </SettingsAccordion>}

        <GroupLabel>Assets &amp; Engines</GroupLabel>
        {isAdmin && <SettingsAccordion
          title="Optional Krea / GGUF Assets"
          summary={
            <Stack direction="row" spacing={0.75}>
              <Chip size="small" color="success" label={`${qualityInstalled}/${qualityItems.length || 0} local`} />
              {qualityDownloadableMissing > 0 && <Chip size="small" color="default" label={`${qualityDownloadableMissing} optional missing`} />}
            </Stack>
          }
        >
          <Stack spacing={1.5}>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              Download workflow companion files only when you choose to use them. Missing optional items are not blockers for the default native Turbo workflow.
            </Typography>
            <Stack spacing={1}>
              {(qualityAssets?.items ?? []).map(asset => (
                <Box key={asset.id} sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 1.5, p: 1 }}>
                  <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" alignItems={{ xs: 'stretch', sm: 'center' }} gap={1}>
                    <Box>
                      <Typography variant="body2">{asset.purpose}</Typography>
                      <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', wordBreak: 'break-all' }}>
                        {asset.repo_id}{asset.filename ? ` · ${asset.filename}` : ''} · {asset.local_path}
                      </Typography>
                      {asset.disabled_reason && (
                        <Typography variant="caption" sx={{ color: 'warning.main', display: 'block' }}>
                          {asset.disabled_reason}
                        </Typography>
                      )}
                    </Box>
                    <Stack direction="row" spacing={1} alignItems="center">
                      <Chip size="small" label={asset.installed ? 'Installed' : asset.download_enabled ? 'Optional' : 'Blocked'} color={asset.installed ? 'success' : asset.download_enabled ? 'default' : 'warning'} />
                      <Button
                        size="small"
                        variant="outlined"
                        onClick={() => downloadQualityAsset(asset.id)}
                        disabled={qualityBusy === asset.id || !asset.download_enabled || asset.installed}
                        startIcon={qualityBusy === asset.id ? <CircularProgress size={14} color="inherit" /> : undefined}
                      >
                        {qualityBusy === asset.id ? 'Downloading...' : asset.installed ? 'Present' : 'Download'}
                      </Button>
                    </Stack>
                  </Stack>
                </Box>
              ))}
            </Stack>
            {qualityMessage && <Alert severity={qualityMessage.severity} sx={{ py: 0 }}>{qualityMessage.text}</Alert>}
          </Stack>
        </SettingsAccordion>}

        {isAdmin && <SettingsAccordion
          title="Native Low-VRAM Diffusion"
          summary={<Chip size="small" color={settingsDraft.diffusion_engine === 'native_pytorch' ? 'default' : 'primary'} label={settingsDraft.diffusion_engine === 'native_pytorch' ? 'Native PyTorch active' : settingsDraft.diffusion_engine === 'native_gguf' ? 'Native GGUF active' : 'Native INT8 active'} />}
        >
          <Stack spacing={1.25}>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              GGUF and INT8 are loaded in-process through Krea's native sampler, Qwen conditioning, LoRA, moodboards, and the selected native VAE mode. No stable-diffusion.cpp sidecar is used.
            </Typography>
            <Button
              variant="contained"
              color="primary"
              size="small"
              onClick={setupGgufLowVram}
              disabled={ggufRuntimeBusy}
              startIcon={ggufRuntimeBusy ? <CircularProgress size={14} color="inherit" /> : undefined}
              sx={{ alignSelf: 'flex-start' }}
            >
              {ggufRuntimeBusy ? 'Setting up Native GGUF...' : 'Setup Native GGUF Low-VRAM'}
            </Button>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              {(['native_pytorch', 'native_gguf', 'native_int8_convrot'] as const).map(engine => (
                <Chip
                  key={engine}
                  label={engine === 'native_pytorch' ? 'Native PyTorch' : engine === 'native_gguf' ? 'Native GGUF' : 'Native INT8 ConvRot'}
                  clickable
                  variant={settingsDraft.diffusion_engine === engine ? 'filled' : 'outlined'}
                  color={settingsDraft.diffusion_engine === engine ? (engine === 'native_int8_convrot' ? 'warning' : 'primary') : 'default'}
                  onClick={() => setSettingsDraft(d => ({ ...d, diffusion_engine: engine }))}
                />
              ))}
            </Stack>
            <TextField label="Krea2 Turbo GGUF path" size="small" fullWidth value={settingsDraft.gguf_turbo_path} onChange={e => setSettingsDraft(d => ({ ...d, gguf_turbo_path: e.target.value }))} placeholder="models\\gguf\\Krea-2-Turbo-Q4_K_M.gguf" />
            <TextField label="Krea2 RAW GGUF path (optional)" size="small" fullWidth value={settingsDraft.gguf_raw_path} onChange={e => setSettingsDraft(d => ({ ...d, gguf_raw_path: e.target.value }))} />
            <Typography variant="subtitle2" sx={{ pt: 1 }}>Native INT8 ConvRot</Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              Ported Krea2 INT8 ConvRot loader. Uses torch._int_mm first; comfy_kitchen/Triton are optional later and are not required.
            </Typography>
            <TextField label="Krea2 Turbo INT8 ConvRot path" size="small" fullWidth value={settingsDraft.krea2_turbo_int8_path} onChange={e => setSettingsDraft(d => ({ ...d, krea2_turbo_int8_path: e.target.value }))} placeholder="models\\krea2\\diffusion_models\\krea2_turbo_int8_convrot.safetensors" />
            <TextField label="Krea2 RAW INT8 ConvRot path (optional)" size="small" fullWidth value={settingsDraft.krea2_raw_int8_path} onChange={e => setSettingsDraft(d => ({ ...d, krea2_raw_int8_path: e.target.value }))} placeholder="models\\krea2\\diffusion_models\\krea2_raw_int8_convrot.safetensors" />
            <Button
              variant="contained"
              color="warning"
              size="small"
              onClick={setupNativeInt8}
              disabled={ggufRuntimeBusy}
              startIcon={ggufRuntimeBusy ? <CircularProgress size={14} color="inherit" /> : undefined}
              sx={{ alignSelf: 'flex-start' }}
            >
              Setup Native INT8
            </Button>
          </Stack>
        </SettingsAccordion>}

        <GroupLabel>Sharing &amp; Admin</GroupLabel>
        {isAdmin && <SettingsAccordion
          title="Users"
          summary={<Chip size="small" label={`${users.length} account${users.length === 1 ? '' : 's'}`} />}
        >
          <Stack spacing={1.5}>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              Admins can manage sharing, settings, models, passwords, safety review, and all galleries. Users can generate normally. Child accounts generate with safety moderation and a private gallery. Only admins can see this list; the dot shows who's online (green) or generating (amber).
            </Typography>
            <Stack spacing={1}>
              {users.map(user => (
                <Stack key={user.username} direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" alignItems={{ xs: 'stretch', sm: 'center' }} gap={1}>
                  <Stack direction="row" spacing={1} alignItems="center">
                    <Tooltip title={user.active ? 'Generating now' : user.online ? 'Online (active in last 2 min)' : 'Offline'}>
                      <Box sx={{ width: 9, height: 9, borderRadius: '50%', flexShrink: 0,
                        bgcolor: user.active ? '#ffb300' : user.online ? '#66bb6a' : 'rgba(202,196,208,0.35)',
                        boxShadow: user.active ? '0 0 6px #ffb300' : user.online ? '0 0 5px #66bb6a' : 'none' }} />
                    </Tooltip>
                    <Typography variant="body2">{user.username}</Typography>
                    {user.active && <Chip size="small" label="working" sx={{ height: 18, fontSize: 10 }} color="warning" variant="outlined" />}
                    {!user.active && user.online && <Chip size="small" label="online" sx={{ height: 18, fontSize: 10 }} color="success" variant="outlined" />}
                  </Stack>
                  <Stack direction="row" spacing={1}>
                    {(['child', 'user', 'admin'] as const).map(role => (
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
              <TextField select SelectProps={{ native: true }} label="Role" size="small" value={newUser.role} onChange={e => setNewUser(u => ({ ...u, role: e.target.value as 'admin' | 'user' | 'child' }))} sx={{ minWidth: 120 }}>
                <option value="child">child</option>
                <option value="user">user</option>
                <option value="admin">admin</option>
              </TextField>
              <Button variant="contained" size="small" onClick={addUser} disabled={!newUser.username || newUser.password.length < 8}>
                Add
              </Button>
            </Stack>
            {userMessage && <Alert severity={userMessage.severity} sx={{ py: 0 }}>{userMessage.text}</Alert>}
          </Stack>
        </SettingsAccordion>}

        {isAdmin && <SettingsAccordion
          title="Child Safety Review"
          summary={<Chip size="small" color={moderationStatus?.image_classifier_available ? 'success' : 'warning'} label={moderationStatus?.image_classifier_available ? 'Classifier ready' : 'Classifier missing'} />}
        >
          <Stack spacing={1.5}>
            <Stack direction="row" justifyContent="space-between" alignItems="center">
              <Button size="small" variant="outlined" onClick={loadModerationEvents} disabled={moderationBusy}>
                {moderationBusy ? 'Refreshing…' : 'Refresh'}
              </Button>
            </Stack>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              Child prompt/image blocks are recorded here for admin review. Quarantined images are admin-only and never shown in a child gallery.
            </Typography>
            {moderationStatus && (
              <Alert severity={moderationStatus.image_classifier_available ? 'success' : 'warning'} sx={{ py: 0 }}>
                {moderationStatus.message}
              </Alert>
            )}
            {!moderationStatus?.image_classifier_available && (
              <Button
                size="small"
                variant="outlined"
                onClick={installImageClassifier}
                disabled={moderationInstallBusy}
                sx={{ alignSelf: 'flex-start' }}
              >
                {moderationInstallBusy ? 'Setting up classifier…' : 'Set up child image classifier'}
              </Button>
            )}
            {moderationEvents.length === 0 ? (
              <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                No moderation events yet.
              </Typography>
            ) : (
              <Stack spacing={1}>
                {moderationEvents.slice(0, 12).map(event => (
                  <Box key={event.id} sx={{ p: 1, border: '1px solid rgba(255,255,255,0.08)', borderRadius: 2 }}>
                    <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" gap={1}>
                      <Box sx={{ minWidth: 0 }}>
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          {event.username} · {event.action.replace(/_/g, ' ')} · {event.mode || event.event_type}
                        </Typography>
                        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>
                          {event.created_at} · {event.reason || 'No reason recorded'}
                        </Typography>
                        {event.prompt && (
                          <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mt: 0.5, wordBreak: 'break-word' }}>
                            Prompt: {event.prompt.slice(0, 240)}
                          </Typography>
                        )}
                      </Box>
                      {event.quarantined_filename && (
                        <Button
                          size="small"
                          variant="outlined"
                          href={publicUrl(`/api/moderation/quarantine/${encodeURIComponent(event.quarantined_filename)}`)}
                          target="_blank"
                          rel="noreferrer"
                        >
                          View quarantine
                        </Button>
                      )}
                    </Stack>
                  </Box>
                ))}
              </Stack>
            )}
          </Stack>
        </SettingsAccordion>}

        {isAdmin && <SettingsAccordion
          title="Tailscale Sharing"
          summary={<Chip size="small" label={sharing?.funnel.running ? 'Sharing' : 'Stopped'} color={sharing?.funnel.running ? 'success' : 'default'} />}
        >
          <Stack spacing={1.5}>
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
              <Button size="small" variant="outlined" onClick={repairSharing} disabled={sharingBusy || !sharing?.tailscale.installed}>
                Repair /krea Sharing
              </Button>
              <Button size="small" color="error" variant="outlined" onClick={stopSharing} disabled={sharingBusy || !sharing?.tailscale.installed}>
                Stop /krea Funnel
              </Button>
            </Stack>
          </Stack>
        </SettingsAccordion>}

        {(report?.attention_acceleration || report?.variants.length) && (
          <SettingsAccordion
            title="Runtime Diagnostics"
            summary={<Chip size="small" label={`${report?.variants.filter(v => v.ok).length ?? 0}/${report?.variants.length ?? 0} variants OK`} color={report?.variants.every(v => v.ok) ? 'success' : 'warning'} />}
          >
            <Stack spacing={1}>
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
          </SettingsAccordion>
        )}
      </Stack>

      <Dialog open={!!resetTarget} onClose={() => setResetTarget(null)} maxWidth="xs" fullWidth>
        <DialogTitle>Reset password{resetTarget ? ` for ${resetTarget}` : ''}</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus fullWidth type="password" label="New password" sx={{ mt: 1 }}
            value={resetPasswordValue}
            onChange={e => setResetPasswordValue(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && resetPasswordValue) runResetPassword() }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setResetTarget(null)}>Cancel</Button>
          <Button variant="contained" onClick={runResetPassword} disabled={!resetPasswordValue}>Reset password</Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}
