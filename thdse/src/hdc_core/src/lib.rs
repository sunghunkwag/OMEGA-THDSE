use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyIndexError};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use std::ptr;

/// Zero-cost abstraction Memory Arena for complex FHRR hypervectors built for PyO3.
///
/// Supports dynamic dimension expansion (Meta-Grammar Emergence) and
/// algebraic fusion operators (bind ∘ bundle) for topological phase transitions.
#[pyclass]
pub struct FhrrArena {
    buffer: Vec<f32>,
    dimension: usize,
    capacity: usize,
    head: usize,
    /// Tracks cumulative bind/bundle operations per handle for thermodynamic cost.
    op_counts: Vec<(u32, u32)>,  // (bind_count, bundle_count) per handle
}

#[pymethods]
impl FhrrArena {
    #[new]
    pub fn new(capacity: usize, dimension: usize) -> Self {
        let floats_per_vec = dimension * 2;
        let total_floats = capacity * floats_per_vec;
        
        let mut buffer = Vec::with_capacity(total_floats);
        
        unsafe {
            ptr::write_bytes(buffer.as_mut_ptr(), 0, total_floats);
            buffer.set_len(total_floats);
        }

        let op_counts = vec![(0u32, 0u32); capacity];

        Self {
            buffer,
            dimension,
            capacity,
            head: 0,
            op_counts,
        }
    }

    pub fn allocate(&mut self) -> PyResult<usize> {
        if self.head >= self.capacity {
            return Err(PyValueError::new_err("Arena capacity exhausted."));
        }
        let id = self.head;
        self.head += 1;
        Ok(id)
    }

    pub fn reset(&mut self) {
        self.head = 0;
    }

    pub fn inject_phases(&mut self, handle: usize, phases: Vec<f32>) -> PyResult<()> {
        if handle >= self.head { return Err(PyIndexError::new_err("Invalid Handle")); }
        if phases.len() != self.dimension { return Err(PyValueError::new_err("Phase length mismatch")); }
        
        let offset = handle * self.dimension * 2;
        let p_data = self.buffer.as_mut_ptr();
        unsafe {
            for (i, &theta) in phases.iter().enumerate() {
                let (sin_t, cos_t) = theta.sin_cos();
                *p_data.add(offset + 2 * i) = cos_t;         
                *p_data.add(offset + 2 * i + 1) = sin_t;     
            }
        }
        Ok(())
    }

    pub fn bind(&mut self, h1: usize, h2: usize, out_handle: usize) -> PyResult<()> {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe { self.bind_simd(h1, h2, out_handle)?; }
            } else {
                self.bind_scalar(h1, h2, out_handle)?;
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.bind_scalar(h1, h2, out_handle)?;
        }
        // Track thermodynamic cost: each bind adds ln(2) entropy
        if out_handle < self.capacity {
            self.op_counts[out_handle].0 += 1;
        }
        Ok(())
    }

    pub fn bundle(&mut self, handles: Vec<usize>, out_handle: usize) -> PyResult<()> {
        let fan_in = handles.len() as u32;
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { self.bundle_simd(handles, out_handle)?; }
            } else {
                self.bundle_scalar(handles, out_handle)?;
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.bundle_scalar(handles, out_handle)?;
        }
        // Track thermodynamic cost: bundle of k handles adds k×ln(2) entropy
        if out_handle < self.capacity {
            self.op_counts[out_handle].1 += fan_in;
        }
        Ok(())
    }

    pub fn compute_correlation(&self, h1: usize, h2: usize) -> PyResult<f32> {
        self.validate_handles(&[h1, h2])?;
        let floats = self.dimension * 2;
        let p_buf = self.buffer.as_ptr();
        let mut sum_real = 0.0;

        unsafe {
            let mut p1 = p_buf.add(h1 * floats);
            let mut p2 = p_buf.add(h2 * floats);
            let end = p1.add(floats);

            while p1 < end {
                let re1 = *p1; let im1 = *p1.add(1);
                let re2 = *p2; let im2 = *p2.add(1);

                sum_real += re1 * re2 + im1 * im2;

                p1 = p1.add(2);
                p2 = p2.add(2);
            }
        }
        Ok(sum_real / (self.dimension as f32))
    }

    // ── Meta-Grammar Emergence: Dimension & Operator Expansion ───

    /// Returns the current hypervector dimension.
    pub fn get_dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the current number of allocated handles.
    pub fn get_head(&self) -> usize {
        self.head
    }

    /// Returns the arena capacity.
    pub fn get_capacity(&self) -> usize {
        self.capacity
    }

    /// Deterministically expand the arena dimension from d to new_dim (new_dim > d).
    ///
    /// For each existing vector, the original d components are preserved.
    /// New components [d, new_dim) are derived deterministically:
    ///   new[j] = conjugate(existing[j % d])  (phase negation = algebraic reflection)
    ///
    /// This is O(head × new_dim) — linear in the number of live vectors.
    pub fn expand_dimension(&mut self, new_dim: usize) -> PyResult<()> {
        if new_dim <= self.dimension {
            return Err(PyValueError::new_err(
                "New dimension must be strictly greater than current dimension."
            ));
        }

        let old_dim = self.dimension;
        let old_floats = old_dim * 2;
        let new_floats = new_dim * 2;
        let total_new = self.capacity * new_floats;

        let mut new_buffer: Vec<f32> = Vec::with_capacity(total_new);
        unsafe {
            ptr::write_bytes(new_buffer.as_mut_ptr(), 0, total_new);
            new_buffer.set_len(total_new);
        }

        // Copy and extend each live vector
        for h in 0..self.head {
            let old_offset = h * old_floats;
            let new_offset = h * new_floats;

            // Copy original components
            unsafe {
                ptr::copy_nonoverlapping(
                    self.buffer.as_ptr().add(old_offset),
                    new_buffer.as_mut_ptr().add(new_offset),
                    old_floats,
                );
            }

            // Deterministic extension: conjugate reflection of existing components
            // new[j] for j in [old_dim, new_dim): (re_{j%d}, -im_{j%d})
            for j in old_dim..new_dim {
                let src_j = j % old_dim;
                unsafe {
                    let src_re = *self.buffer.as_ptr().add(old_offset + src_j * 2);
                    let src_im = *self.buffer.as_ptr().add(old_offset + src_j * 2 + 1);
                    *new_buffer.as_mut_ptr().add(new_offset + j * 2) = src_re;
                    *new_buffer.as_mut_ptr().add(new_offset + j * 2 + 1) = -src_im;
                }
            }
        }

        self.buffer = new_buffer;
        self.dimension = new_dim;
        Ok(())
    }

    /// Algebraic fusion operator: out = bind(h_bind, bundle(handles))
    ///
    /// This is a new topological grammar rule synthesized from ⊗ and ⊕:
    ///   fuse(A, {B₁,...,Bₖ}) = A ⊗ (B₁ ⊕ ... ⊕ Bₖ)
    ///
    /// Executed as a single fused pass to avoid intermediate allocation.
    /// The bundle is computed into a temporary region, then bound with h_bind.
    pub fn bind_bundle_fusion(
        &mut self, h_bind: usize, handles_bundle: Vec<usize>, out_handle: usize,
    ) -> PyResult<()> {
        if handles_bundle.is_empty() {
            return Err(PyValueError::new_err("Bundle handles must be non-empty."));
        }
        self.validate_handles(&[h_bind, out_handle])?;
        self.validate_handles(&handles_bundle)?;

        let floats = self.dimension * 2;

        // Step 1: Compute bundle into out_handle (temporary use)
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { self.bundle_simd(handles_bundle, out_handle)?; }
            } else {
                self.bundle_scalar(handles_bundle, out_handle)?;
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.bundle_scalar(handles_bundle, out_handle)?;
        }

        // Step 2: Bind h_bind with bundled result in-place
        // We need a temporary copy of the bundle result
        let bundle_copy: Vec<f32> = self.buffer
            [out_handle * floats..(out_handle + 1) * floats]
            .to_vec();

        // Perform complex multiply: out = h_bind ⊗ bundle_result
        let p_buf = self.buffer.as_mut_ptr();
        unsafe {
            let p_bind = p_buf.add(h_bind * floats);
            let p_out = p_buf.add(out_handle * floats);

            for i in 0..self.dimension {
                let re1 = *p_bind.add(i * 2);
                let im1 = *p_bind.add(i * 2 + 1);
                let re2 = bundle_copy[i * 2];
                let im2 = bundle_copy[i * 2 + 1];
                *p_out.add(i * 2) = re1 * re2 - im1 * im2;
                *p_out.add(i * 2 + 1) = re1 * im2 + im1 * re2;
            }
        }

        // Track op costs: fusion = 1 bind + 1 bundle
        self.op_counts[out_handle].0 += 1;
        self.op_counts[out_handle].1 += 1;
        Ok(())
    }

    // ── Topological Thermodynamics: Operation Cost Tracking ──────

    /// Record a bind operation on the output handle.
    pub fn record_bind_cost(&mut self, handle: usize) -> PyResult<()> {
        if handle >= self.head {
            return Err(PyIndexError::new_err("Invalid handle"));
        }
        self.op_counts[handle].0 += 1;
        Ok(())
    }

    /// Record a bundle operation on the output handle.
    pub fn record_bundle_cost(&mut self, handle: usize, fan_in: usize) -> PyResult<()> {
        if handle >= self.head {
            return Err(PyIndexError::new_err("Invalid handle"));
        }
        self.op_counts[handle].1 += fan_in as u32;
        Ok(())
    }

    /// Retrieve the thermodynamic cost (bind_count, bundle_count) for a handle.
    pub fn get_op_counts(&self, handle: usize) -> PyResult<(u32, u32)> {
        if handle >= self.head {
            return Err(PyIndexError::new_err("Invalid handle"));
        }
        Ok(self.op_counts[handle])
    }

    /// Compute the thermodynamic entropy penalty for a handle.
    ///
    /// S(h) = bind_count × ln(2) + bundle_count × ln(bundle_fan_in)
    ///
    /// This is a deterministic, O(1) complexity metric.
    pub fn compute_entropy(&self, handle: usize) -> PyResult<f32> {
        if handle >= self.head {
            return Err(PyIndexError::new_err("Invalid handle"));
        }
        let (binds, bundles) = self.op_counts[handle];
        let ln2: f32 = 2.0_f32.ln();
        // Each bind costs ln(2), each bundle fan-in costs ln(fan_in) ≈ ln(2) per unit
        let entropy = (binds as f32) * ln2 + (bundles as f32) * ln2;
        Ok(entropy)
    }

    // ── Dynamic Null Space Projection: Quotient Space Folding ─────

    /// Project the entire arena into the quotient space H / <V_error>.
    ///
    /// For every live hypervector X in the arena, annihilate the component
    /// parallel to the Contradiction Vector V_error:
    ///
    ///   X_new = X - V_error · (V_error^H · X) / (V_error^H · V_error)
    ///
    /// where V_error^H denotes the conjugate transpose (Hermitian adjoint).
    ///
    /// After projection, each component is re-normalized to unit magnitude
    /// to preserve the FHRR invariant (|z_j| = 1 for all j).
    ///
    /// This maps the vector space H → H / <V_error>, collapsing the axis
    /// of contradiction to zero and forcing previously conflicting structures
    /// to become mathematically equivalent.
    ///
    /// Complexity: O(head × dimension) — linear in live vectors × dimension.
    /// Memory: in-place modification, no auxiliary arena allocation.
    pub fn project_to_quotient_space(&mut self, v_error_id: usize) -> PyResult<usize> {
        self.validate_handles(&[v_error_id])?;
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { self.project_to_quotient_space_simd(v_error_id) };
            }
        }
        self.project_to_quotient_space_scalar(v_error_id)
    }

    /// Project the arena into the quotient space H / <V₁, V₂, ..., Vₙ>.
    ///
    /// Unlike single-vector projection, this removes the ENTIRE subspace
    /// spanned by the given vectors, not just one axis at a time.
    ///
    /// Algorithm:
    ///   1. Gram-Schmidt orthogonalize the V_error vectors
    ///      (ensures sequential projection is mathematically equivalent
    ///       to simultaneous subspace removal)
    ///   2. For each orthogonalized basis vector, project it out from
    ///      all live arena vectors (reusing project_to_quotient_space_simd)
    ///
    /// This is O(k² × d + k × head × d) where k = number of walls.
    ///
    /// Args:
    ///   v_error_ids: Vec<usize> — arena handles of all V_error vectors
    ///
    /// Returns:
    ///   Number of vectors modified.
    pub fn project_to_multi_quotient_space(
        &mut self, v_error_ids: Vec<usize>,
    ) -> PyResult<usize> {
        if v_error_ids.is_empty() {
            return Ok(0);
        }
        self.validate_handles(&v_error_ids)?;

        let floats = self.dimension * 2;
        let k = v_error_ids.len();

        // Step 1: Copy V_error vectors to temporary workspace for Gram-Schmidt
        let mut workspace: Vec<Vec<f32>> = Vec::with_capacity(k);
        for &vid in &v_error_ids {
            let offset = vid * floats;
            workspace.push(self.buffer[offset..offset + floats].to_vec());
        }

        // Step 2: Gram-Schmidt orthogonalization using complex inner products
        // For i in 1..k: for j in 0..i: V_i -= V_j * (<V_j, V_i> / <V_j, V_j>)
        // then normalize V_i
        for i in 1..k {
            for j in 0..i {
                // Complex inner product <V_j, V_i> = V_j^H · V_i
                // = Σ_d conj(V_j[d]) · V_i[d]
                let mut dot_re: f32 = 0.0;
                let mut dot_im: f32 = 0.0;
                let mut norm_sq: f32 = 0.0;

                for d in 0..self.dimension {
                    let vj_re = workspace[j][2 * d];
                    let vj_im = workspace[j][2 * d + 1];
                    let vi_re = workspace[i][2 * d];
                    let vi_im = workspace[i][2 * d + 1];

                    // <V_j, V_i> = conj(V_j) · V_i
                    dot_re += vj_re * vi_re + vj_im * vi_im;
                    dot_im += vj_re * vi_im - vj_im * vi_re;

                    // ||V_j||² = conj(V_j) · V_j (always real)
                    norm_sq += vj_re * vj_re + vj_im * vj_im;
                }

                if norm_sq < 1e-12 {
                    continue; // Skip near-zero basis vector
                }

                // Projection coefficient: c = <V_j, V_i> / ||V_j||²
                let inv_norm_sq = 1.0 / norm_sq;
                let coeff_re = dot_re * inv_norm_sq;
                let coeff_im = dot_im * inv_norm_sq;

                // V_i -= V_j * c (complex scalar-vector multiply)
                for d in 0..self.dimension {
                    let vj_re = workspace[j][2 * d];
                    let vj_im = workspace[j][2 * d + 1];
                    // V_j * c: (vj_re * c_re - vj_im * c_im, vj_re * c_im + vj_im * c_re)
                    let proj_re = vj_re * coeff_re - vj_im * coeff_im;
                    let proj_im = vj_re * coeff_im + vj_im * coeff_re;
                    workspace[i][2 * d] -= proj_re;
                    workspace[i][2 * d + 1] -= proj_im;
                }
            }

            // Normalize V_i to unit magnitude per component
            for d in 0..self.dimension {
                let re = workspace[i][2 * d];
                let im = workspace[i][2 * d + 1];
                let mag = (re * re + im * im).sqrt();
                if mag > 1e-12 {
                    let inv_mag = 1.0 / mag;
                    workspace[i][2 * d] = re * inv_mag;
                    workspace[i][2 * d + 1] = im * inv_mag;
                }
            }
        }

        // Step 3: Inject orthogonalized basis vectors into temporary arena slots
        // and project each one out sequentially
        let mut total_projected: usize = 0;

        for i in 0..k {
            // Check if this basis vector is non-zero
            let mut norm_sq: f32 = 0.0;
            for d in 0..self.dimension {
                let re = workspace[i][2 * d];
                let im = workspace[i][2 * d + 1];
                norm_sq += re * re + im * im;
            }
            if norm_sq < 1e-12 {
                continue; // Linearly dependent — skip
            }

            // Allocate a temporary handle for the orthogonalized vector
            let temp_h = self.allocate()?;

            // Inject the orthogonalized vector
            let temp_offset = temp_h * floats;
            self.buffer[temp_offset..temp_offset + floats]
                .copy_from_slice(&workspace[i]);

            // Project it out from all arena vectors
            let count = self.project_to_quotient_space(temp_h)?;
            total_projected += count;
        }

        Ok(total_projected)
    }

    /// Compute the upper triangle of the pairwise correlation matrix
    /// for a list of handles. Returns a flat Vec<f32> of length N*(N-1)/2
    /// in row-major upper-triangle order:
    ///   [(0,1), (0,2), ..., (0,N-1), (1,2), ..., (N-2,N-1)]
    ///
    /// Each correlation is the standard FHRR cosine: Re(V_a^H · V_b) / d.
    ///
    /// This replaces N*(N-1)/2 individual Python→Rust round trips with
    /// a single call. The algorithm is still O(N²·d) but executes entirely
    /// in Rust with cache-friendly sequential memory access.
    pub fn correlate_matrix(&self, handles: Vec<usize>) -> PyResult<Vec<f32>> {
        self.validate_handles(&handles)?;
        let n = handles.len();
        if n < 2 {
            return Ok(Vec::new());
        }
        let num_pairs = n * (n - 1) / 2;
        let mut result = Vec::with_capacity(num_pairs);
        let floats = self.dimension * 2;
        let inv_d = 1.0 / (self.dimension as f32);
        let p_buf = self.buffer.as_ptr();

        for i in 0..n {
            let offset_i = handles[i] * floats;
            for j in (i + 1)..n {
                let offset_j = handles[j] * floats;
                let corr = unsafe {
                    self.correlate_pair_inner(p_buf, offset_i, offset_j, floats, inv_d)
                };
                result.push(corr);
            }
        }

        Ok(result)
    }

    /// Compute correlations between every (query, target) pair.
    /// Returns a flat Vec<f32> of length len(queries) * len(targets)
    /// in row-major order: [q0·t0, q0·t1, ..., q0·tM, q1·t0, ...].
    ///
    /// queries and targets may overlap. Self-correlation pairs are
    /// NOT excluded — the caller decides what to do with them.
    pub fn correlate_matrix_subset(
        &self, queries: Vec<usize>, targets: Vec<usize>,
    ) -> PyResult<Vec<f32>> {
        self.validate_handles(&queries)?;
        self.validate_handles(&targets)?;
        let nq = queries.len();
        let nt = targets.len();
        if nq == 0 || nt == 0 {
            return Ok(Vec::new());
        }
        let mut result = Vec::with_capacity(nq * nt);
        let floats = self.dimension * 2;
        let inv_d = 1.0 / (self.dimension as f32);
        let p_buf = self.buffer.as_ptr();

        for i in 0..nq {
            let offset_q = queries[i] * floats;
            for j in 0..nt {
                let offset_t = targets[j] * floats;
                let corr = unsafe {
                    self.correlate_pair_inner(p_buf, offset_q, offset_t, floats, inv_d)
                };
                result.push(corr);
            }
        }

        Ok(result)
    }

    /// Extract the raw phase array (angles) from a handle. Used by Python
    /// to read back projected vectors without round-tripping through inject.
    pub fn extract_phases(&self, handle: usize) -> PyResult<Vec<f32>> {
        if handle >= self.head {
            return Err(PyIndexError::new_err("Invalid handle"));
        }
        let floats = self.dimension * 2;
        let offset = handle * floats;
        let mut phases = Vec::with_capacity(self.dimension);
        for j in 0..self.dimension {
            let re = self.buffer[offset + 2 * j];
            let im = self.buffer[offset + 2 * j + 1];
            phases.push(im.atan2(re));
        }
        Ok(phases)
    }
}

impl FhrrArena {
    // ── Scalar fallback implementations (safe, no SIMD required) ────

    fn bind_scalar(&mut self, h1: usize, h2: usize, out_handle: usize) -> PyResult<()> {
        self.validate_handles(&[h1, h2, out_handle])?;
        let floats = self.dimension * 2;
        for i in 0..self.dimension {
            let idx1 = h1 * floats + 2 * i;
            let idx2 = h2 * floats + 2 * i;
            let idx_out = out_handle * floats + 2 * i;
            let re1 = self.buffer[idx1];
            let im1 = self.buffer[idx1 + 1];
            let re2 = self.buffer[idx2];
            let im2 = self.buffer[idx2 + 1];
            self.buffer[idx_out] = re1 * re2 - im1 * im2;
            self.buffer[idx_out + 1] = re1 * im2 + im1 * re2;
        }
        Ok(())
    }

    fn bundle_scalar(&mut self, handles: Vec<usize>, out_handle: usize) -> PyResult<()> {
        if handles.is_empty() { return Ok(()); }
        self.validate_handles(&handles)?;
        self.validate_handles(&[out_handle])?;
        let floats = self.dimension * 2;
        let out_start = out_handle * floats;
        // Zero the output region
        for i in 0..floats {
            self.buffer[out_start + i] = 0.0;
        }
        // Accumulate all handles
        for h in &handles {
            let h_start = h * floats;
            for i in 0..floats {
                self.buffer[out_start + i] += self.buffer[h_start + i];
            }
        }
        // Normalize each complex component to unit magnitude
        for i in 0..self.dimension {
            let idx = out_start + 2 * i;
            let re = self.buffer[idx];
            let im = self.buffer[idx + 1];
            let mag = (re * re + im * im).sqrt();
            if mag > 1e-12 {
                let inv_mag = 1.0 / mag;
                self.buffer[idx] = re * inv_mag;
                self.buffer[idx + 1] = im * inv_mag;
            }
        }
        Ok(())
    }

    fn project_to_quotient_space_scalar(&mut self, v_error_id: usize) -> PyResult<usize> {
        let floats = self.dimension * 2;
        let v_offset = v_error_id * floats;

        // Compute ||V_error||²
        let mut norm_sq: f32 = 0.0;
        for i in 0..floats {
            let val = self.buffer[v_offset + i];
            norm_sq += val * val;
        }
        if norm_sq < 1e-12 {
            return Ok(0);
        }
        let inv_norm_sq = 1.0 / norm_sq;
        let mut projected_count: usize = 0;

        for h in 0..self.head {
            if h == v_error_id { continue; }
            let x_offset = h * floats;

            // Pass 1: Complex inner product <V_error, X>
            let mut dot_re: f32 = 0.0;
            let mut dot_im: f32 = 0.0;
            for j in 0..self.dimension {
                let v_re = self.buffer[v_offset + 2 * j];
                let v_im = self.buffer[v_offset + 2 * j + 1];
                let x_re = self.buffer[x_offset + 2 * j];
                let x_im = self.buffer[x_offset + 2 * j + 1];
                dot_re += v_re * x_re + v_im * x_im;
                dot_im += v_re * x_im - v_im * x_re;
            }

            let coeff_re = dot_re * inv_norm_sq;
            let coeff_im = dot_im * inv_norm_sq;
            if coeff_re * coeff_re + coeff_im * coeff_im < 1e-16 {
                continue;
            }

            // Pass 2: X -= V · c, then normalize
            for j in 0..self.dimension {
                let v_re = self.buffer[v_offset + 2 * j];
                let v_im = self.buffer[v_offset + 2 * j + 1];
                let proj_re = v_re * coeff_re - v_im * coeff_im;
                let proj_im = v_re * coeff_im + v_im * coeff_re;
                self.buffer[x_offset + 2 * j] -= proj_re;
                self.buffer[x_offset + 2 * j + 1] -= proj_im;
            }

            // Normalize to unit magnitude
            for j in 0..self.dimension {
                let idx = x_offset + 2 * j;
                let re = self.buffer[idx];
                let im = self.buffer[idx + 1];
                let mag = (re * re + im * im).sqrt();
                if mag > 1e-12 {
                    let inv_mag = 1.0 / mag;
                    self.buffer[idx] = re * inv_mag;
                    self.buffer[idx + 1] = im * inv_mag;
                }
            }
            projected_count += 1;
        }
        Ok(projected_count)
    }

    // ── AVX2 SIMD implementations (guarded by runtime detection) ────

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn bind_simd(&mut self, h1: usize, h2: usize, out_handle: usize) -> PyResult<()> {
        self.validate_handles(&[h1, h2, out_handle])?;
        
        let floats = self.dimension * 2;
        let p_buf = self.buffer.as_mut_ptr();
        
        let mut p1 = p_buf.add(h1 * floats);
        let mut p2 = p_buf.add(h2 * floats);
        let mut p_out = p_buf.add(out_handle * floats);
        let end = p1.add(floats);
        
        let simd_end = end.sub(floats % 8);
        while p1 < simd_end {
            let a = _mm256_loadu_ps(p1);
            let b = _mm256_loadu_ps(p2);

            let a_real = _mm256_moveldup_ps(a);
            let a_imag = _mm256_movehdup_ps(a);

            let b_shuf = _mm256_shuffle_ps(b, b, 0xb1);

            let res1 = _mm256_mul_ps(a_real, b);
            let res2 = _mm256_mul_ps(a_imag, b_shuf);

            let result = _mm256_addsub_ps(res1, res2);

            _mm256_storeu_ps(p_out, result);

            p1 = p1.add(8);
            p2 = p2.add(8);
            p_out = p_out.add(8);
        }
        
        while p1 < end {
            let re1 = *p1; let im1 = *p1.add(1);
            let re2 = *p2; let im2 = *p2.add(1);
            *p_out = re1 * re2 - im1 * im2;
            *p_out.add(1) = re1 * im2 + im1 * re2;
            p1 = p1.add(2);
            p2 = p2.add(2);
            p_out = p_out.add(2);
        }

        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn bundle_simd(&mut self, handles: Vec<usize>, out_handle: usize) -> PyResult<()> {
        if handles.is_empty() { return Ok(()); }
        self.validate_handles(&handles)?;
        self.validate_handles(&[out_handle])?;
        
        let floats = self.dimension * 2;
        let p_buf = self.buffer.as_mut_ptr();
        let p_out = p_buf.add(out_handle * floats);

        ptr::write_bytes(p_out, 0, floats);

        for h in handles {
            let mut p_curr = p_buf.add(h * floats);
            let mut p_out_curr = p_out;
            let end = p_curr.add(floats);
            let simd_end = end.sub(floats % 8);

            while p_curr < simd_end {
                let v_out = _mm256_loadu_ps(p_out_curr);
                let v_curr = _mm256_loadu_ps(p_curr);
                let summed = _mm256_add_ps(v_out, v_curr);
                _mm256_storeu_ps(p_out_curr, summed);
                p_curr = p_curr.add(8);
                p_out_curr = p_out_curr.add(8);
            }

            while p_curr < end {
                *p_out_curr += *p_curr;
                p_curr = p_curr.add(1);
                p_out_curr = p_out_curr.add(1);
            }
        }

        self.normalize_simd(out_handle)?;
        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn normalize_simd(&mut self, handle: usize) -> PyResult<()> {
        let floats = self.dimension * 2;
        let p_buf = self.buffer.as_mut_ptr();
        let mut p_vec = p_buf.add(handle * floats);
        let end = p_vec.add(floats);

        let simd_end = end.sub(floats % 8);
        while p_vec < simd_end {
            let v = _mm256_loadu_ps(p_vec);
            let sq = _mm256_mul_ps(v, v);
            
            let shuf = _mm256_shuffle_ps(sq, sq, 0xb1);
            let mags_sq = _mm256_add_ps(sq, shuf);
            
            let inv_mags = _mm256_rsqrt_ps(mags_sq);
            let normalized = _mm256_mul_ps(v, inv_mags);
            
            _mm256_storeu_ps(p_vec, normalized);
            p_vec = p_vec.add(8);
        }

        while p_vec < end {
            let re = *p_vec;
            let im = *p_vec.add(1);
            let mag_inv = 1.0 / (re*re + im*im).sqrt();
            if mag_inv.is_finite() {
                *p_vec *= mag_inv;
                *p_vec.add(1) *= mag_inv;
            }
            p_vec = p_vec.add(2);
        }
        Ok(())
    }

    /// Inner correlation computation for a single pair.
    /// Computes Re(V_a^H · V_b) / d using scalar loop.
    /// Called from correlate_matrix and correlate_matrix_subset.
    ///
    /// # Safety
    /// Caller must ensure offset_a and offset_b are valid buffer offsets
    /// and that p_buf points to the arena buffer with sufficient length.
    #[inline]
    unsafe fn correlate_pair_inner(
        &self, p_buf: *const f32,
        offset_a: usize, offset_b: usize,
        floats: usize, inv_d: f32,
    ) -> f32 {
        let mut p1 = p_buf.add(offset_a);
        let mut p2 = p_buf.add(offset_b);
        let end = p1.add(floats);
        let mut sum_real: f32 = 0.0;

        while p1 < end {
            let re1 = *p1;
            let im1 = *p1.add(1);
            let re2 = *p2;
            let im2 = *p2.add(1);
            sum_real += re1 * re2 + im1 * im2;
            p1 = p1.add(2);
            p2 = p2.add(2);
        }

        sum_real * inv_d
    }

    fn validate_handles(&self, handles: &[usize]) -> PyResult<()> {
        for &h in handles {
            if h >= self.head {
                return Err(PyIndexError::new_err(format!("Handle {} out of bounds.", h)));
            }
        }
        Ok(())
    }

    /// AVX2-accelerated quotient space projection.
    ///
    /// Two-pass algorithm per vector X:
    ///   Pass 1 (scalar): compute complex inner product <V_error, X> and ||V_error||²
    ///   Pass 2 (SIMD):   X -= V_error · (<V_error^H · X> / ||V_error||²), then normalize
    ///
    /// Returns the number of vectors projected.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn project_to_quotient_space_simd(&mut self, v_error_id: usize) -> PyResult<usize> {
        let floats = self.dimension * 2;
        let v_offset = v_error_id * floats;

        // Compute ||V_error||² = V_error^H · V_error (always real)
        let mut norm_sq: f32 = 0.0;
        for i in 0..floats {
            let val = self.buffer[v_offset + i];
            norm_sq += val * val;
        }

        if norm_sq < 1e-12 {
            // V_error is effectively zero — nothing to project out
            return Ok(0);
        }

        let inv_norm_sq = 1.0 / norm_sq;
        let mut projected_count: usize = 0;

        for h in 0..self.head {
            if h == v_error_id {
                continue;
            }

            let x_offset = h * floats;

            // ── Pass 1: Complex inner product <V_error, X> = V^H · X ──
            // = Σ_j conj(V_j) · X_j
            // = Σ_j (V_re·X_re + V_im·X_im) + i·(V_re·X_im − V_im·X_re)
            let mut dot_re: f32 = 0.0;
            let mut dot_im: f32 = 0.0;

            for j in 0..self.dimension {
                let v_re = *self.buffer.as_ptr().add(v_offset + 2 * j);
                let v_im = *self.buffer.as_ptr().add(v_offset + 2 * j + 1);
                let x_re = *self.buffer.as_ptr().add(x_offset + 2 * j);
                let x_im = *self.buffer.as_ptr().add(x_offset + 2 * j + 1);

                dot_re += v_re * x_re + v_im * x_im;
                dot_im += v_re * x_im - v_im * x_re;
            }

            // Projection coefficient: c = <V^H, X> / ||V||²
            let coeff_re = dot_re * inv_norm_sq;
            let coeff_im = dot_im * inv_norm_sq;

            // Skip if projection component is negligible
            if coeff_re * coeff_re + coeff_im * coeff_im < 1e-16 {
                continue;
            }

            // ── Pass 2 (AVX2): X -= V · c, then normalize ─────────────
            // V·c component j:
            //   proj_re = V_re·c_re − V_im·c_im
            //   proj_im = V_re·c_im + V_im·c_re
            //
            // We broadcast c_re to even lanes, c_im to odd lanes, then
            // use the same addsub pattern as bind_simd.

            // Build broadcast vectors for coefficient
            // coeff_interleaved = [c_re, c_im, c_re, c_im, ...]
            let coeff_arr: [f32; 8] = [
                coeff_re, coeff_im, coeff_re, coeff_im,
                coeff_re, coeff_im, coeff_re, coeff_im,
            ];
            let v_coeff = _mm256_loadu_ps(coeff_arr.as_ptr());
            // coeff_swapped = [c_im, c_re, c_im, c_re, ...]
            let v_coeff_swap = _mm256_shuffle_ps(v_coeff, v_coeff, 0xb1);

            let p_buf = self.buffer.as_mut_ptr();
            let mut p_v = p_buf.add(v_offset);
            let mut p_x = p_buf.add(x_offset);
            let end = p_v.add(floats);
            let simd_end = end.sub(floats % 8);

            while p_v < simd_end {
                let v_vec = _mm256_loadu_ps(p_v);
                let x_vec = _mm256_loadu_ps(p_x);

                // Complex multiply V · coeff using moveldup/movehdup + addsub
                let v_real = _mm256_moveldup_ps(v_vec);  // [v_re, v_re, ...]
                let v_imag = _mm256_movehdup_ps(v_vec);  // [v_im, v_im, ...]

                let prod1 = _mm256_mul_ps(v_real, v_coeff);      // [v_re·c_re, v_re·c_im, ...]
                let prod2 = _mm256_mul_ps(v_imag, v_coeff_swap); // [v_im·c_im, v_im·c_re, ...]

                // addsub: even lanes subtract, odd lanes add
                // result = [v_re·c_re − v_im·c_im, v_re·c_im + v_im·c_re, ...]
                let proj = _mm256_addsub_ps(prod1, prod2);

                // X -= proj
                let result = _mm256_sub_ps(x_vec, proj);
                _mm256_storeu_ps(p_x, result);

                p_v = p_v.add(8);
                p_x = p_x.add(8);
            }

            // Scalar tail for remaining elements
            while p_v < end {
                let v_re = *p_v;
                let v_im = *p_v.add(1);
                let proj_re = v_re * coeff_re - v_im * coeff_im;
                let proj_im = v_re * coeff_im + v_im * coeff_re;
                *p_x -= proj_re;
                *p_x.add(1) -= proj_im;
                p_v = p_v.add(2);
                p_x = p_x.add(2);
            }

            // Re-normalize each component to unit magnitude (FHRR invariant)
            self.normalize_simd(h)?;

            projected_count += 1;
        }

        Ok(projected_count)
    }
}

#[pymodule]
fn hdc_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FhrrArena>()?;
    Ok(())
}
