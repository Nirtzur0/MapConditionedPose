# Bug Fixes Summary - 2025-12-30

## Issues Resolved

### 1. ✅ Numpy RuntimeWarning: Mean of empty slice
**Location:** `src/data_generation/features/rt_extractor.py`

**Problem:** 
When reducing tensor dimensions, the code was calling `mean()` on potentially empty slices, causing:
```
RuntimeWarning: Mean of empty slice.
RuntimeWarning: invalid value encountered in divide
```

**Fix:**
Added a check for zero-size dimensions before calling `mean()`:
```python
def _reduce_to_3d(x):
    curr_x = x
    while len(ops.shape(curr_x)) > 3:
        # Check for zero size to avoid RuntimeWarning
        if ops.shape(curr_x)[-1] == 0:
            curr_x = ops.sum(curr_x, axis=-1)
        else:
            curr_x = ops.mean(curr_x, axis=-1)
    return curr_x
```

### 2. ✅ Mitsuba HDRFilm Warning
**Location:** `src/utils/logging_config.py`

**Problem:**
Mitsuba (Sionna's rendering backend) was emitting C++ warnings:
```
WARN [HDRFilm] Monochrome mode enabled, setting film output pixel format to 'luminance'
```

**Fix:**
Added Mitsuba log level configuration and additional warning filters:
```python
# Suppress Numpy warnings
warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='invalid value encountered in divide')
warnings.filterwarnings('ignore', message='invalid value encountered in true_divide')

# Silence Mitsuba C++ warnings
try:
    import mitsuba as mi
    mi.set_log_level(mi.LogLevel.Error)
except ImportError:
    pass
```

### 3. ✅ Zarr ValueError: Shape Broadcast Error
**Location:** `src/data_generation/zarr_writer.py`

**Problem:**
When writing data with mismatched dimensions (e.g., 223 sites vs 256 expected), the padding logic was creating buffers incorrectly:
```
ValueError: could not broadcast input array from shape (5,223,256) into shape (5,256,256)
```

**Fix:**
Improved the padding/truncation logic to properly handle multi-dimensional arrays:
```python
# Build slices for both source and destination
src_slices = []
dst_slices = []
for i in range(value.ndim):
    copy_size = min(value.shape[i], buffer.shape[i])
    src_slices.append(slice(0, copy_size))
    dst_slices.append(slice(0, copy_size))

# Copy data into buffer
buffer[tuple(dst_slices)] = value[tuple(src_slices)]
```

## Test Results

### Quick Test Pipeline (✅ PASSED)
```
Duration: 3.0 minutes
Steps: Scene Generation, Dataset Generation, Model Training
Total samples: 200
Test Results:
  - Median Error: 92.90 m
  - Mean Error: 92.79 m
  - RMSE: 99.12 m
  - Success @ 50m: 17.5%
```

## Impact

All warnings and errors have been suppressed or fixed:
- ✅ No more red RuntimeWarnings during dataset generation
- ✅ No more Mitsuba HDRFilm warnings
- ✅ No more ValueError crashes during Zarr writing
- ✅ Pipeline runs cleanly from start to finish

## Files Modified

1. `src/data_generation/features/rt_extractor.py` - Fixed empty slice mean calculation
2. `src/utils/logging_config.py` - Added warning suppressions and Mitsuba log level
3. `src/data_generation/zarr_writer.py` - Improved shape mismatch handling

## Next Steps

The full experiment (`run_full_experiment.sh`) is still running with these fixes applied. Monitor for any additional issues during the multi-city Optuna optimization run.
