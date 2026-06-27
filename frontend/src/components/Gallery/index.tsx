import { useCallback, useEffect, useState } from 'react'
import { Box, CircularProgress, Fab, Grid, IconButton, Stack, Tab, Tabs, Tooltip, Typography } from '@mui/material'
import FavoriteIcon from '@mui/icons-material/Favorite'
import FavoriteBorderIcon from '@mui/icons-material/FavoriteBorder'
import DeleteIcon from '@mui/icons-material/Delete'
import DownloadIcon from '@mui/icons-material/Download'
import RefreshIcon from '@mui/icons-material/Refresh'
import { apiFetch, publicUrl, type GalleryItem } from '../../api'
import { useStore } from '../../store'

export default function GalleryPanel() {
  const [items, setItems] = useState<GalleryItem[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [favoritesOnly, setFavoritesOnly] = useState(false)
  const [loading, setLoading] = useState(false)
  const { openLightbox } = useStore()

  const load = useCallback(async (pg = 1, favs = favoritesOnly) => {
    setLoading(true)
    try {
      const data = await apiFetch.gallery(pg, 50, favs)
      setItems(prev => pg === 1 ? data.items : [...prev, ...data.items])
      setTotal(data.total)
      setPage(pg)
    } finally { setLoading(false) }
  }, [favoritesOnly])

  useEffect(() => { load(1) }, [favoritesOnly])

  useEffect(() => {
    const onUpdated = (event: Event) => {
      const detail = (event as CustomEvent<{ id: number; favorite: boolean }>).detail
      setItems(prev => prev.map(i => i.id === detail.id ? { ...i, favorite: detail.favorite } : i))
    }
    const onDeleted = (event: Event) => {
      const detail = (event as CustomEvent<{ id: number }>).detail
      setItems(prev => prev.filter(i => i.id !== detail.id))
      setTotal(t => Math.max(0, t - 1))
    }
    window.addEventListener('krea-gallery-item-updated', onUpdated)
    window.addEventListener('krea-gallery-item-deleted', onDeleted)
    return () => {
      window.removeEventListener('krea-gallery-item-updated', onUpdated)
      window.removeEventListener('krea-gallery-item-deleted', onDeleted)
    }
  }, [])

  const toggleFav = async (item: GalleryItem) => {
    await apiFetch.setFavorite(item.id, !item.favorite)
    setItems(prev => prev.map(i => i.id === item.id ? { ...i, favorite: !i.favorite } : i))
  }

  const deleteItem = async (id: number) => {
    await apiFetch.deleteGalleryItem(id)
    setItems(prev => prev.filter(i => i.id !== id))
    setTotal(t => t - 1)
  }

  const openAt = (idx: number) => {
    openLightbox(items.map(item => ({
      src: publicUrl(`/api/outputs/${item.filename}`),
      id: item.id,
      filename: item.filename,
      prompt: item.prompt,
      favorite: item.favorite,
      metadata: item.metadata,
    })), idx)
  }

  return (
    <Box sx={{ p: { xs: 1.5, sm: 2 } }}>
      <Stack direction="row" justifyContent="space-between" alignItems="center" mb={2}>
        <Tabs value={favoritesOnly ? 1 : 0} onChange={(_, v) => setFavoritesOnly(v === 1)}>
          <Tab label="All" />
          <Tab label="Favorites" />
        </Tabs>
        <Tooltip title="Refresh">
          <IconButton onClick={() => load(1, favoritesOnly)}><RefreshIcon /></IconButton>
        </Tooltip>
      </Stack>

      {loading && page === 1 ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
          <CircularProgress />
        </Box>
      ) : items.length === 0 ? (
        <Box sx={{ textAlign: 'center', py: 8 }}>
          <Typography sx={{ color: 'text.secondary' }}>No images yet. Generate something!</Typography>
        </Box>
      ) : (
        <Grid container spacing={1}>
          {items.map(item => (
            <Grid item xs={6} sm={4} md={3} key={item.id}>
              <Box
                sx={{
                  position: 'relative', borderRadius: 2, overflow: 'hidden',
                  bgcolor: 'background.paper', cursor: 'pointer',
                  '&:hover .actions': { opacity: 1 },
                  aspectRatio: `${item.width} / ${item.height}`,
                }}
              >
                {item.thumbnail_b64 ? (
                  <img
                    src={`data:image/webp;base64,${item.thumbnail_b64}`}
                    alt={item.prompt.slice(0, 40)}
                    style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                    onClick={() => openAt(items.findIndex(i => i.id === item.id))}
                  />
                ) : (
                  <Box sx={{ width: '100%', height: '100%', bgcolor: 'background.paper', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Typography variant="caption" sx={{ color: 'text.disabled' }}>No preview</Typography>
                  </Box>
                )}
                <Stack
                  className="actions"
                  direction="row"
                  sx={{
                    position: 'absolute', bottom: 4, right: 4,
                    opacity: 0, transition: 'opacity 0.2s',
                    bgcolor: 'rgba(0,0,0,0.65)', borderRadius: 1.5,
                  }}
                >
                  <IconButton size="small" onClick={(e) => { e.stopPropagation(); toggleFav(item) }}>
                    {item.favorite ? <FavoriteIcon fontSize="small" sx={{ color: '#F48FB1' }} /> : <FavoriteBorderIcon fontSize="small" />}
                  </IconButton>
                  <IconButton size="small" component="a" href={publicUrl(`/api/outputs/${item.filename}`)} download onClick={e => e.stopPropagation()}>
                    <DownloadIcon fontSize="small" />
                  </IconButton>
                  <IconButton size="small" onClick={(e) => { e.stopPropagation(); deleteItem(item.id) }} sx={{ color: 'error.light' }}>
                    <DeleteIcon fontSize="small" />
                  </IconButton>
                </Stack>
              </Box>
            </Grid>
          ))}
        </Grid>
      )}

      {items.length < total && (
        <Box sx={{ textAlign: 'center', mt: 2 }}>
          <Fab variant="extended" size="small" onClick={() => load(page + 1)} disabled={loading}>
            {loading ? <CircularProgress size={16} /> : 'Load more'}
          </Fab>
        </Box>
      )}
    </Box>
  )
}
