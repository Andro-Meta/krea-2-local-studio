import React, { useEffect, useState } from 'react'
import { Alert, Box, Button, Chip, CircularProgress, FormControlLabel, LinearProgress, MenuItem, Paper, Stack, Switch, TextField, Typography } from '@mui/material'
import GpuIcon from '@mui/icons-material/Memory'
import { apiFetch, publicUrl, type AcceleratorStatus, type AppSettings, type AuthSession, type KreaServerProcess, type ModerationEvent, type ModerationStatus, type QualityAsset, type ShareUser, type SharingStatus, type SystemReport } from '../../api'
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
    openrouter_api_key: '',
    openrouter_model: 'google/gemma-4-31b-it:free',
    openrouter_free_only: true,
    krea_attention_backend: 'sdpa' as 'sdpa' | 'sage',
    local_llm_backend: 'transformers' as 'transformers' | 'gguf_server',
    local_qwen_model_id: '',
    gguf_helper_base_url: 'http://127.0.0.1:1234/v1',
    gguf_helper_model: 'BennyDaBall/Krea-2-Engineer-V1-GGUF:Q4_K_M',
    gguf_helper_timeout_sec: 120,
    diffusion_engine: 'native_pytorch' as 'native_pytorch' | 'native_int8_convrot' | 'gguf_external' | 'int8_convrot_external',
    krea2_turbo_int8_path: '',
    krea2_raw_int8_path: '',
    gguf_sd_cli_path: '',
    gguf_turbo_path: '',
    gguf_raw_path: '',
    gguf_llm_path: '',
    gguf_vae_path: '',
    gguf_lora_dir: '',
    gguf_timeout_sec: 600,
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
  const { setSystemReport, setRealtimeSettings, setParams } = useStore()
  const isAdmin = auth?.role === 'admin'
  const localQwenChoice = !settingsDraft.local_qwen_model_id
    ? 'default'
    : /Huihui-Qwen3-VL-4B-Instruct-abliterated|qwen3_vl_4b_abliterated/i.test(settingsDraft.local_qwen_model_id)
      ? 'abliterated'
      : 'custom'

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
        openrouter_api_key: '',
        openrouter_model: s.openrouter_model,
        openrouter_free_only: s.openrouter_free_only,
        krea_attention_backend: s.krea_attention_backend ?? 'sdpa',
        local_llm_backend: s.local_llm_backend ?? 'transformers',
        local_qwen_model_id: s.local_qwen_model_id ?? '',
        gguf_helper_base_url: s.gguf_helper_base_url ?? 'http://127.0.0.1:1234/v1',
        gguf_helper_model: s.gguf_helper_model ?? 'BennyDaBall/Krea-2-Engineer-V1-GGUF:Q4_K_M',
        gguf_helper_timeout_sec: s.gguf_helper_timeout_sec ?? 120,
        diffusion_engine: s.diffusion_engine ?? 'native_pytorch',
        krea2_turbo_int8_path: s.krea2_turbo_int8_path ?? '',
        krea2_raw_int8_path: s.krea2_raw_int8_path ?? '',
        gguf_sd_cli_path: s.gguf_sd_cli_path ?? '',
        gguf_turbo_path: s.gguf_turbo_path ?? '',
        gguf_raw_path: s.gguf_raw_path ?? '',
        gguf_llm_path: s.gguf_llm_path ?? '',
        gguf_vae_path: s.gguf_vae_path ?? '',
        gguf_lora_dir: s.gguf_lora_dir ?? '',
        gguf_timeout_sec: s.gguf_timeout_sec ?? 600,
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
    loadAuth().then(session => {
      if (session?.role === 'admin') {
        loadSettings()
        loadUsers()
        loadSharing()
        loadQualityAssets()
        loadModerationEvents()
        loadAccelerators()
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
        local_qwen_model_id: settingsDraft.local_qwen_model_id,
        gguf_helper_base_url: settingsDraft.gguf_helper_base_url,
        gguf_helper_model: settingsDraft.gguf_helper_model,
        gguf_helper_timeout_sec: settingsDraft.gguf_helper_timeout_sec,
        diffusion_engine: settingsDraft.diffusion_engine,
        gguf_sd_cli_path: settingsDraft.gguf_sd_cli_path,
        gguf_turbo_path: settingsDraft.gguf_turbo_path,
        gguf_raw_path: settingsDraft.gguf_raw_path,
        gguf_llm_path: settingsDraft.gguf_llm_path,
        gguf_vae_path: settingsDraft.gguf_vae_path,
        gguf_lora_dir: settingsDraft.gguf_lora_dir,
        gguf_timeout_sec: settingsDraft.gguf_timeout_sec,
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

  const testGgufRuntime = async () => {
    setGgufRuntimeBusy(true)
    setSettingsMessage(null)
    try {
      await apiFetch.updateSettings({
        diffusion_engine: settingsDraft.diffusion_engine,
        krea2_turbo_int8_path: settingsDraft.krea2_turbo_int8_path,
        krea2_raw_int8_path: settingsDraft.krea2_raw_int8_path,
        gguf_sd_cli_path: settingsDraft.gguf_sd_cli_path,
        gguf_turbo_path: settingsDraft.gguf_turbo_path,
        gguf_raw_path: settingsDraft.gguf_raw_path,
        gguf_llm_path: settingsDraft.gguf_llm_path,
        gguf_vae_path: settingsDraft.gguf_vae_path,
        gguf_lora_dir: settingsDraft.gguf_lora_dir,
        gguf_timeout_sec: settingsDraft.gguf_timeout_sec,
      })
      const result = await apiFetch.testGgufRuntime()
      setSettingsMessage({ severity: 'success', text: `GGUF runtime dry-run OK: ${result.command.slice(0, 5).join(' ')}` })
      await loadSettings()
    } catch (e: any) {
      setSettingsMessage({ severity: 'error', text: e?.response?.data?.detail ?? e.message ?? 'GGUF runtime test failed.' })
    } finally {
      setGgufRuntimeBusy(false)
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
        gguf_sd_cli_path: result.sd_cli_path,
        gguf_turbo_path: result.turbo_path,
        gguf_llm_path: result.llm_path,
        gguf_vae_path: result.vae_path,
      }))
      setRealtimeSettings({
        previewSize: result.realtime.preview_size,
        previewSteps: result.realtime.preview_steps,
        finalSteps: result.realtime.final_steps,
      })
      setParams({
        diffusion_engine: 'gguf_external',
        model_profile: '',
        mode: 'txt2img',
        checkpoint: 'turbo',
        quantization: 'fp8',
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
        loras: [],
        style_references: [],
        regional_prompts: [],
        moodboard_images: [],
        selected_moodboard_ids: [],
        moodboard_uuids: [],
        use_rebalance: false,
        krea_enhancer_enabled: false,
        krea_enhancer_variant: 'off',
        cfg_zero_star: false,
        conditioning_mode: 'auto',
        negative_prompt: '',
      })
      await loadQualityAssets()
      await loadSettings()
      setSettingsMessage({
        severity: 'warning',
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

        {isAdmin && <Paper sx={{ p: 2 }}>
          <Stack spacing={1.25}>
            <Stack direction="row" justifyContent="space-between" alignItems="center">
              <Typography variant="h6">Experimental Accelerators</Typography>
              <Button size="small" variant="text" onClick={loadAccelerators}>Refresh</Button>
            </Stack>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              PyTorch SDPA is the stable default. Triton/SageAttention are opt-in experiments; they may cause black/noisy outputs, so verify visually before keeping them enabled.
            </Typography>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              <Chip size="small" color="success" label="SDPA default" />
              <Chip size="small" color={accelerators?.triton_windows.installed ? 'success' : 'default'} label={`Triton ${accelerators?.triton_windows.installed ? 'installed' : 'not installed'}`} />
              <Chip size="small" color={accelerators?.sageattention.installed ? 'success' : 'default'} label={`Sage ${accelerators?.sageattention.installed ? 'installed' : 'not installed'}`} />
            </Stack>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              <Button
                size="small"
                variant="outlined"
                disabled={!!acceleratorBusy}
                startIcon={acceleratorBusy === 'triton' ? <CircularProgress size={14} /> : undefined}
                onClick={() => installAccelerator('triton')}
              >
                Install Triton for Windows
              </Button>
              <Button
                size="small"
                variant="outlined"
                disabled={!!acceleratorBusy}
                startIcon={acceleratorBusy === 'sage' ? <CircularProgress size={14} /> : undefined}
                onClick={() => installAccelerator('sage')}
              >
                Install SageAttention
              </Button>
            </Stack>
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={settingsDraft.krea_attention_backend === 'sage'}
                  disabled={!!acceleratorBusy || !accelerators?.sageattention.installed}
                  onChange={e => saveAttentionBackend(e.target.checked ? 'sage' : 'sdpa')}
                />
              }
              label={
                <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                  Enable SageAttention for A/B test{!accelerators?.sageattention.installed ? ' — install SageAttention first' : ''}
                </Typography>
              }
            />
            {acceleratorMessage && <Alert severity={acceleratorMessage.severity} sx={{ py: 0 }}>{acceleratorMessage.text}</Alert>}
          </Stack>
        </Paper>}

        {/* GPU profile + per-system recommendation */}
        {report?.gpu_capabilities && (
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" mb={1}>GPU Profile</Typography>
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
          </Paper>
        )}

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
                  label="VAE path"
                  value={vaePath}
                  onChange={e => setVaePath(e.target.value)}
                  size="small"
                  fullWidth
                  disabled={!isAdmin}
                  placeholder="models\krea2\vae\wan_2.1_vae.safetensors"
                  helperText="Default: Wan 2.1 VAE when present. Empty falls back to stock Qwen VAE. Applies on next model load."
                />
                <Button
                  variant="outlined" size="small" sx={{ mt: 0.5 }}
                  disabled={!isAdmin || vaeSaving}
                  onClick={async () => {
                    setVaeSaving(true)
                    try {
                      await apiFetch.updateSettings({ krea2_vae_path: vaePath })
                      setSettingsMessage({ severity: 'success', text: 'VAE path saved. Reload the model to apply.' })
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
              Local Qwen3-VL is the self-contained helper for prompt expansion and image-to-prompt. GGUF helper is text-only and can use LM Studio or llama.cpp. OpenRouter and Ideogram remain optional hosted helpers.
            </Typography>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              {(['transformers', 'gguf_server'] as const).map(b => (
                <Chip
                  key={b}
                  label={b === 'transformers' ? 'Transformers Qwen3-VL' : 'GGUF server'}
                  clickable
                  variant={settingsDraft.local_llm_backend === b ? 'filled' : 'outlined'}
                  color={settingsDraft.local_llm_backend === b ? 'secondary' : 'default'}
                  onClick={() => setSettingsDraft(d => ({ ...d, local_llm_backend: b }))}
                />
              ))}
            </Stack>
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
                  helperText="Xperiment selects Abliterated Qwen by default. The Comfy FP8 abliterated file is not Transformers-loadable; this uses the BF16 Transformers repo or a local downloaded copy."
                >
                  <MenuItem value="default">Default installed Qwen3-VL</MenuItem>
                  <MenuItem value="abliterated">Abliterated Qwen3-VL (Xperiment)</MenuItem>
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
            <Typography variant="h6">Optional Krea / GGUF Assets</Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              Download workflow companion files only when you choose to use them. These are not required for the default native Turbo workflow.
            </Typography>
            <Stack spacing={1}>
              {(qualityAssets?.items.filter(asset => asset.id !== 'flux_fill') ?? []).map(asset => (
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
                        disabled={qualityBusy === asset.id || !asset.download_enabled}
                        startIcon={qualityBusy === asset.id ? <CircularProgress size={14} color="inherit" /> : undefined}
                      >
                        {qualityBusy === asset.id ? 'Downloading...' : 'Download'}
                      </Button>
                    </Stack>
                  </Stack>
                </Box>
              ))}
            </Stack>
            {qualityMessage && <Alert severity={qualityMessage.severity} sx={{ py: 0 }}>{qualityMessage.text}</Alert>}
          </Stack>
        </Paper>}

        {isAdmin && <Paper sx={{ p: 2 }}>
          <Stack spacing={1.25}>
            <Typography variant="h6">GGUF / External Diffusion Runtime</Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              Experimental low-VRAM sidecar path. Native PyTorch remains the default. Configure stable-diffusion.cpp paths here, then use the engine selector in Create.
            </Typography>
            <Button
              variant="contained"
              color="warning"
              size="small"
              onClick={setupGgufLowVram}
              disabled={ggufRuntimeBusy}
              startIcon={ggufRuntimeBusy ? <CircularProgress size={14} color="inherit" /> : undefined}
              sx={{ alignSelf: 'flex-start' }}
            >
              {ggufRuntimeBusy ? 'Setting up GGUF...' : 'Setup GGUF Low-VRAM'}
            </Button>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              {(['native_pytorch', 'native_int8_convrot', 'gguf_external'] as const).map(engine => (
                <Chip
                  key={engine}
                  label={engine === 'native_pytorch' ? 'Native PyTorch' : engine === 'native_int8_convrot' ? 'Native INT8 ConvRot' : 'GGUF external'}
                  clickable
                  variant={settingsDraft.diffusion_engine === engine ? 'filled' : 'outlined'}
                  color={settingsDraft.diffusion_engine === engine ? (engine === 'native_pytorch' ? 'primary' : 'warning') : 'default'}
                  onClick={() => setSettingsDraft(d => ({ ...d, diffusion_engine: engine }))}
                />
              ))}
            </Stack>
            <TextField label="stable-diffusion.cpp sd-cli path" size="small" fullWidth value={settingsDraft.gguf_sd_cli_path} onChange={e => setSettingsDraft(d => ({ ...d, gguf_sd_cli_path: e.target.value }))} placeholder="C:\\tools\\stable-diffusion.cpp\\sd-cli.exe" />
            <TextField label="Krea2 Turbo GGUF path" size="small" fullWidth value={settingsDraft.gguf_turbo_path} onChange={e => setSettingsDraft(d => ({ ...d, gguf_turbo_path: e.target.value }))} placeholder="models\\gguf\\Krea-2-Turbo-Q4_K_M.gguf" />
            <TextField label="Krea2 RAW GGUF path (optional)" size="small" fullWidth value={settingsDraft.gguf_raw_path} onChange={e => setSettingsDraft(d => ({ ...d, gguf_raw_path: e.target.value }))} />
            <TextField label="Qwen3-VL GGUF LLM path" size="small" fullWidth value={settingsDraft.gguf_llm_path} onChange={e => setSettingsDraft(d => ({ ...d, gguf_llm_path: e.target.value }))} />
            <TextField label="VAE path" size="small" fullWidth value={settingsDraft.gguf_vae_path} onChange={e => setSettingsDraft(d => ({ ...d, gguf_vae_path: e.target.value }))} />
            <TextField label="LoRA directory (optional; disabled until A/B verified)" size="small" fullWidth value={settingsDraft.gguf_lora_dir} onChange={e => setSettingsDraft(d => ({ ...d, gguf_lora_dir: e.target.value }))} />
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
            <TextField
              label="GGUF runtime timeout (seconds)"
              type="number"
              size="small"
              value={settingsDraft.gguf_timeout_sec}
              onChange={e => setSettingsDraft(d => ({ ...d, gguf_timeout_sec: Math.max(60, Number(e.target.value) || 600) }))}
              inputProps={{ min: 60, step: 60 }}
            />
            <Button
              variant="outlined"
              size="small"
              onClick={testGgufRuntime}
              disabled={ggufRuntimeBusy}
              startIcon={ggufRuntimeBusy ? <CircularProgress size={14} color="inherit" /> : undefined}
              sx={{ alignSelf: 'flex-start' }}
            >
              Test GGUF Runtime Dry Run
            </Button>
          </Stack>
        </Paper>}

        {isAdmin && <Paper sx={{ p: 2 }}>
          <Stack spacing={1.5}>
            <Typography variant="h6">Users</Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              Admins can manage sharing, settings, models, passwords, safety review, and all galleries. Users can generate normally. Child accounts generate with safety moderation and a private gallery.
            </Typography>
            <Stack spacing={1}>
              {users.map(user => (
                <Stack key={user.username} direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" alignItems={{ xs: 'stretch', sm: 'center' }} gap={1}>
                  <Typography variant="body2">{user.username}</Typography>
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
        </Paper>}

        {isAdmin && <Paper sx={{ p: 2 }}>
          <Stack spacing={1.5}>
            <Stack direction="row" justifyContent="space-between" alignItems="center">
              <Typography variant="h6">Child Safety Review</Typography>
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
              <Button size="small" variant="outlined" onClick={repairSharing} disabled={sharingBusy || !sharing?.tailscale.installed}>
                Repair /krea Sharing
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
