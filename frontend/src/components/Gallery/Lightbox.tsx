import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Alert, Box, CircularProgress, IconButton, Modal, Snackbar, Tooltip, Typography } from '@mui/material'
import CloseIcon from '@mui/icons-material/Close'
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft'
import ChevronRightIcon from '@mui/icons-material/ChevronRight'
import DeleteIcon from '@mui/icons-material/Delete'
import DownloadIcon from '@mui/icons-material/Download'
import FavoriteIcon from '@mui/icons-material/Favorite'
import FavoriteBorderIcon from '@mui/icons-material/FavoriteBorder'
import CollectionsIcon from '@mui/icons-material/Collections'
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome'
import ImageSearchIcon from '@mui/icons-material/ImageSearch'
import BrushIcon from '@mui/icons-material/Brush'
import OpenInFullIcon from '@mui/icons-material/OpenInFull'
import { apiFetch } from '../../api'
import { useStore } from '../../store'
import { downloadImage, srcToBase64 } from '../../lib/imageActions'

type Point = { x: number; y: number }

function distance(a: Point, b: Point) {
  return Math.hypot(a.x - b.x, a.y - b.y)
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function eventPoint(e: React.PointerEvent): Point {
  return { x: e.clientX, y: e.clientY }
}

export default function Lightbox() {
  const {
    lightbox, closeLightbox, nextLightbox, previousLightbox, patchLightboxItem,
    removeLightboxItem, params, setParams, setTab,
  } = useStore()
  const item = lightbox?.items[lightbox.index]
  const [scale, setScale] = useState(1)
  const [offset, setOffset] = useState<Point>({ x: 0, y: 0 })
  const [toast, setToast] = useState<{ text: string; severity: 'success' | 'info' | 'error' } | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const pointers = useRef(new Map<number, Point>())
  const lastPan = useRef<Point | null>(null)
  const lastPinchDistance = useRef<number | null>(null)
  const swipeStart = useRef<Point | null>(null)

  const hasMany = (lightbox?.items.length ?? 0) > 1
  const counter = useMemo(() => lightbox ? `${lightbox.index + 1} / ${lightbox.items.length}` : '', [lightbox])

  useEffect(() => {
    setScale(1)
    setOffset({ x: 0, y: 0 })
  }, [item?.src])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeLightbox()
      if (e.key === 'ArrowRight') nextLightbox()
      if (e.key === 'ArrowLeft') previousLightbox()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [closeLightbox, nextLightbox, previousLightbox])

  if (!lightbox || !item) return null

  const resetZoom = () => {
    setScale(1)
    setOffset({ x: 0, y: 0 })
  }

  const zoomBy = (delta: number, center?: Point) => {
    setScale(prev => {
      const next = clamp(prev + delta, 1, 5)
      if (next === 1) setOffset({ x: 0, y: 0 })
      if (center && next > 1) setOffset(o => ({ x: o.x + (center.x - window.innerWidth / 2) * 0.06, y: o.y + (center.y - window.innerHeight / 2) * 0.06 }))
      return next
    })
  }

  const handlePointerDown = (e: React.PointerEvent) => {
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
    pointers.current.set(e.pointerId, eventPoint(e))
    if (pointers.current.size === 1) {
      lastPan.current = eventPoint(e)
      swipeStart.current = eventPoint(e)
    }
    if (pointers.current.size === 2) {
      const pts = Array.from(pointers.current.values())
      lastPinchDistance.current = distance(pts[0], pts[1])
    }
  }

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!pointers.current.has(e.pointerId)) return
    pointers.current.set(e.pointerId, eventPoint(e))
    if (pointers.current.size === 2) {
      const pts = Array.from(pointers.current.values())
      const d = distance(pts[0], pts[1])
      if (lastPinchDistance.current) zoomBy((d - lastPinchDistance.current) / 180)
      lastPinchDistance.current = d
      return
    }
    if (scale > 1 && lastPan.current) {
      const p = eventPoint(e)
      setOffset(o => ({ x: o.x + p.x - lastPan.current!.x, y: o.y + p.y - lastPan.current!.y }))
      lastPan.current = p
    }
  }

  const handlePointerUp = (e: React.PointerEvent) => {
    const start = swipeStart.current
    const end = eventPoint(e)
    pointers.current.delete(e.pointerId)
    lastPan.current = null
    lastPinchDistance.current = null
    if (scale <= 1 && start && Math.abs(end.x - start.x) > 70 && Math.abs(end.y - start.y) < 80) {
      if (end.x < start.x) nextLightbox()
      else previousLightbox()
    }
  }

  const withImage = async (action: (b64: string) => Promise<void> | void) => {
    if (!item) return
    const b64 = await srcToBase64(item.src)
    await action(b64)
  }

  const loadTo = async (mode: 'img2img' | 'inpaint' | 'outpaint') => {
    setBusy('Loading image')
    try {
      await withImage(b64 => setParams({ mode, init_image_b64: b64, mask_b64: '' }))
      setTab(0)
      closeLightbox()
    } catch (e: any) {
      setToast({ text: e?.message ?? 'Could not load image', severity: 'error' })
    } finally {
      setBusy(null)
    }
  }

  const addMoodboard = async () => {
    setBusy('Adding to moodboard')
    try {
      await withImage(b64 => setParams({ moodboard_images: [...params.moodboard_images, b64] }))
      setToast({ text: 'Loaded to moodboard', severity: 'success' })
    } catch (e: any) {
      setToast({ text: e?.message ?? 'Could not add image', severity: 'error' })
    } finally {
      setBusy(null)
    }
  }

  const describe = async () => {
    setBusy('Creating prompt')
    try {
      await withImage(async b64 => {
        const result = await apiFetch.describeImage(b64)
        setParams({ prompt: result.prompt, mode: 'txt2img' })
      })
      setTab(0)
      closeLightbox()
    } catch (e: any) {
      setToast({ text: e?.response?.data?.detail ?? e?.message ?? 'Could not create prompt', severity: 'error' })
    } finally {
      setBusy(null)
    }
  }

  const toggleFavorite = async () => {
    if (!item.id) return
    const favorite = !item.favorite
    await apiFetch.setFavorite(item.id, favorite)
    patchLightboxItem(item.id, { favorite })
    window.dispatchEvent(new CustomEvent('krea-gallery-item-updated', { detail: { id: item.id, favorite } }))
  }

  const deleteCurrent = async () => {
    if (!item.id || !window.confirm('Delete this image from the gallery?')) return
    await apiFetch.deleteGalleryItem(item.id)
    removeLightboxItem(item.id)
    window.dispatchEvent(new CustomEvent('krea-gallery-item-deleted', { detail: { id: item.id } }))
  }

  const actionSx = { bgcolor: 'rgba(0,0,0,0.55)', minWidth: 44, minHeight: 44 }

  return (
    <Modal open onClose={closeLightbox}>
      <Box
        sx={{
          position: 'fixed', inset: 0, display: 'flex', alignItems: 'center',
          justifyContent: 'center', bgcolor: 'rgba(0,0,0,0.9)', overflow: 'hidden',
          touchAction: 'none', userSelect: 'none',
        }}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onWheel={e => zoomBy(-e.deltaY / 500, { x: e.clientX, y: e.clientY })}
        onDoubleClick={() => scale > 1 ? resetZoom() : setScale(2)}
      >
        <Typography sx={{ position: 'fixed', top: 18, left: 18, color: 'rgba(255,255,255,0.8)', zIndex: 2 }}>
          {counter}
        </Typography>
        <img
          src={item.src} alt={item.prompt || 'Full size'}
          draggable={false}
          style={{
            maxWidth: '100vw',
            maxHeight: '100vh',
            objectFit: 'contain',
            borderRadius: 8,
            transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
            transition: pointers.current.size ? 'none' : 'transform 120ms ease-out',
            cursor: scale > 1 ? 'grab' : 'zoom-in',
          }}
          onClick={e => e.stopPropagation()}
        />
        {hasMany && (
          <>
            <IconButton sx={{ position: 'fixed', left: { xs: 8, sm: 18 }, top: '50%', ...actionSx }} onClick={previousLightbox}>
              <ChevronLeftIcon />
            </IconButton>
            <IconButton sx={{ position: 'fixed', right: { xs: 8, sm: 18 }, top: '50%', ...actionSx }} onClick={nextLightbox}>
              <ChevronRightIcon />
            </IconButton>
          </>
        )}
        <IconButton
          sx={{ position: 'fixed', top: 16, right: 16, ...actionSx }}
          onClick={closeLightbox}
        >
          <CloseIcon />
        </IconButton>
        {busy && (
          <Box sx={{ position: 'fixed', inset: 0, display: 'grid', placeItems: 'center', pointerEvents: 'none' }}>
            <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', bgcolor: 'rgba(0,0,0,0.72)', px: 2, py: 1.5, borderRadius: 2 }}>
              <CircularProgress size={18} color="inherit" />
              <Typography>{busy}...</Typography>
            </Box>
          </Box>
        )}
        <Box
          sx={{
            position: 'fixed', left: 0, right: 0, bottom: 0, p: 1,
            display: 'flex', gap: 0.75, justifyContent: 'center', flexWrap: 'wrap',
            bgcolor: 'linear-gradient(transparent, rgba(0,0,0,0.75))',
          }}
          onPointerDown={e => e.stopPropagation()}
          onClick={e => e.stopPropagation()}
        >
          {item.id && (
            <Tooltip title="Favorite">
              <IconButton sx={actionSx} onClick={toggleFavorite}>
                {item.favorite ? <FavoriteIcon sx={{ color: '#F48FB1' }} /> : <FavoriteBorderIcon />}
              </IconButton>
            </Tooltip>
          )}
          <Tooltip title="Download">
            <IconButton sx={actionSx} onClick={() => downloadImage(item.src, item.filename)}>
              <DownloadIcon />
            </IconButton>
          </Tooltip>
          {item.id && (
            <Tooltip title="Delete">
              <IconButton sx={{ ...actionSx, color: 'error.light' }} onClick={deleteCurrent}>
                <DeleteIcon />
              </IconButton>
            </Tooltip>
          )}
          <Tooltip title="Load to moodboard">
            <IconButton sx={actionSx} onClick={addMoodboard}><CollectionsIcon /></IconButton>
          </Tooltip>
          <Tooltip title="Create prompt from image">
            <IconButton sx={actionSx} onClick={describe}><AutoAwesomeIcon /></IconButton>
          </Tooltip>
          <Tooltip title="Load to image to image">
            <IconButton sx={actionSx} onClick={() => loadTo('img2img')}><ImageSearchIcon /></IconButton>
          </Tooltip>
          <Tooltip title="Load to inpaint">
            <IconButton sx={actionSx} onClick={() => loadTo('inpaint')}><BrushIcon /></IconButton>
          </Tooltip>
          <Tooltip title="Load to outpaint">
            <IconButton sx={actionSx} onClick={() => loadTo('outpaint')}><OpenInFullIcon /></IconButton>
          </Tooltip>
        </Box>
        <Snackbar open={!!toast} autoHideDuration={5000} onClose={() => setToast(null)}>
          <Alert severity={toast?.severity ?? 'info'} variant="filled" onClose={() => setToast(null)}>{toast?.text}</Alert>
        </Snackbar>
      </Box>
    </Modal>
  )
}
