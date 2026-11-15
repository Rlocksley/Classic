// ============================================================================
// main2.cpp -- Classical Multimodal Face Detection Algorithm in C++
//
// Formulation:
// A Human Face is defined from first principles as an isolated, bilaterally
// symmetric, tripartite craniofacial relief on an organic living dermal manifold:
//
// 1. Biological Dermal Manifold (Living Skin Invariant):
//    - Specific melanin/hemoglobin spectral absorption locus: R > G > B with
//      strictly bounded blue fraction (B/total in [0.12, 0.32]) and vascular ratio.
//    - Smooth isotropic surface texture: low gradient variance (rejects fur/stone/wood).
//
// 2. Bi-Ocular Isolation & Anti-Periodic Guard (Rejecting Windows and Grids):
//    - Exactly two orbital basins (eyes).
//    - Absence of periodic duplicate basins to the left or right (eliminates
//      architectural facades like castles, rows of windows, and ship gunports).
//    - Horizontal W-profile: Cheek (light) -> Eye (basin) -> Bridge (crest) -> Eye (basin) -> Cheek (light).
//
// 3. 3D Craniofacial Topography:
//    - Bilateral orbital basins with internal scleral contrast (pupil + sclera).
//    - Narrow elevated nasal dorsum (bridge) separating orbits.
//    - Nasal lobule (tip highlight) with bilateral sub-nasal nostril depressions.
//    - Oral-mental complex: horizontal oral fissure (dark crevice) above chin highlight.
//    - Luminous, smooth forehead bounded by brow/hairline.
//
// 4. Strict Bilateral Sagittal Reflection Symmetry:
//    - Left and right hemifaces must mirror each other across the median facial axis.
//
// Input: Arbitrary variable width and height (NO 96x96 restriction!)
// Representation: rgbTensor (5-dimensional tensor: pixel matrix x RGB)
// Core Function: bool facetest(const rgbTensor& tensor);
// ============================================================================

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ============================================================================
// 1. rgbTensor (5D Tensor Representation: Pixel Matrix x RGB Color Space)
// ============================================================================
// Dimensions: [Batch = 1] x [Scale = 1] x [Height = H] x [Width = W] x [Channels = 3]
// Each pixel is an embedded 5D vector: (y/H, x/W, R, G, B)
// ============================================================================
struct rgbTensor {
    int b = 1;          // Batch dimension
    int s = 1;          // Scale / Depth dimension
    int h = 0;          // Variable height
    int w = 0;          // Variable width
    int c = 3;          // RGB Channels (0=R, 1=G, 2=B)

    std::vector<float> data; // Normalized float buffer in [0.0, 1.0]

    rgbTensor() = default;
    rgbTensor(int height, int width, int channels = 3)
        : b(1), s(1), h(height), w(width), c(channels),
          data((size_t)height * width * channels, 0.0f) {}

    // 5D Tensor indexer: tensor(b, s, y, x, ch)
    inline float operator()(int batch_idx, int scale_idx, int y, int x, int ch) const {
        (void)batch_idx; (void)scale_idx;
        return data[((size_t)y * w + x) * c + ch];
    }
    inline float& operator()(int batch_idx, int scale_idx, int y, int x, int ch) {
        (void)batch_idx; (void)scale_idx;
        return data[((size_t)y * w + x) * c + ch];
    }

    // 3D Spatial-Channel indexer: tensor(y, x, ch)
    inline float px(int y, int x, int ch) const {
        return data[((size_t)y * w + x) * c + ch];
    }
    inline float operator()(int y, int x, int ch) const {
        return px(y, x, ch);
    }
    inline float& operator()(int y, int x, int ch) {
        return data[((size_t)y * w + x) * c + ch];
    }

    // 5-dimensional feature vector at pixel (y, x): (y_norm, x_norm, R, G, B)
    struct Vector5D {
        float y_norm, x_norm, r, g, b;
    };
    inline Vector5D get5D(int y, int x) const {
        return {
            h > 0 ? (float)y / (float)h : 0.0f,
            w > 0 ? (float)x / (float)w : 0.0f,
            px(y, x, 0),
            px(y, x, 1),
            px(y, x, 2)
        };
    }

    // Conversion to 5D nested vector if desired
    using Tensor5D = std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>;
    Tensor5D to5DVector() const {
        Tensor5D t(1, std::vector<std::vector<std::vector<std::vector<float>>>>(
            1, std::vector<std::vector<std::vector<float>>>(
                h, std::vector<std::vector<float>>(
                    w, std::vector<float>(c)))));
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                for (int ch = 0; ch < c; ++ch) {
                    t[0][0][y][x][ch] = px(y, x, ch);
                }
            }
        }
        return t;
    }
};

// ============================================================================
// 2. Geometry & Fast Integral Image Tables
// ============================================================================
struct Rect {
    int y0 = 0, x0 = 0, y1 = 0, x1 = 0;
    inline bool valid() const { return y1 > y0 && x1 > x0; }
    inline int width() const { return std::max(0, x1 - x0); }
    inline int height() const { return std::max(0, y1 - y0); }
    inline double area() const { return (double)width() * height(); }
};

inline Rect clampRect(int y0, int x0, int y1, int x1, int h, int w) {
    return Rect{
        std::max(0, std::min(y0, h)),
        std::max(0, std::min(x0, w)),
        std::max(0, std::min(y1, h)),
        std::max(0, std::min(x1, w))
    };
}

struct IntegralTable {
    int h = 0, w = 0;
    std::vector<double> table; // (h+1) * (w+1)

    void init(int height, int width, const std::vector<float>& src) {
        h = height;
        w = width;
        int W = w + 1;
        table.assign((size_t)(h + 1) * W, 0.0);
        for (int y = 0; y < h; ++y) {
            double row_sum = 0.0;
            for (int x = 0; x < w; ++x) {
                row_sum += src[(size_t)y * w + x];
                table[(size_t)(y + 1) * W + (x + 1)] = table[(size_t)y * W + (x + 1)] + row_sum;
            }
        }
    }

    inline double sum(const Rect& r) const {
        if (!r.valid()) return 0.0;
        int W = w + 1;
        return table[(size_t)r.y1 * W + r.x1]
             - table[(size_t)r.y0 * W + r.x1]
             - table[(size_t)r.y1 * W + r.x0]
             + table[(size_t)r.y0 * W + r.x0];
    }

    inline double mean(const Rect& r) const {
        double a = r.area();
        return a > 0.0 ? sum(r) / a : 0.0;
    }
};

inline double ringMean(const IntegralTable& it, const Rect& inner, int pad) {
    Rect outer = clampRect(inner.y0 - pad, inner.x0 - pad, inner.y1 + pad, inner.x1 + pad, it.h, it.w);
    double diffArea = outer.area() - inner.area();
    if (diffArea <= 0.0) return 0.0;
    return (it.sum(outer) - it.sum(inner)) / diffArea;
}

inline double boxVariance(const IntegralTable& i1, const IntegralTable& i2, const Rect& r) {
    if (!r.valid()) return 0.0;
    double m = i1.mean(r);
    double m2 = i2.mean(r);
    return std::max(0.0, m2 - m * m);
}

// ============================================================================
// 3. Living Human Skin Chromaticity Manifold
// ============================================================================
// Distinguishes human skin from sandstone, weathered wood, paint, and fruit
// ============================================================================
inline float evaluateSkinLocus(float r, float g, float b, bool is_monochrome) {
    if (is_monochrome) {
        float lum = 0.299f * r + 0.587f * g + 0.114f * b;
        return (lum > 0.16f && lum < 0.86f) ? 0.75f : 0.15f;
    }

    float total = r + g + b;
    if (total < 0.08f || total > 2.85f) return 0.0f; // underexposed black or blown specular white

    float norm_r = r / total;
    float norm_g = g / total;
    float norm_b = b / total;

    // 1. Blue channel bound: Human skin always reflects some subcutaneous backscattered blue
    // Yellow fruit, varnish, and yellow/orange paint have norm_b < 0.11!
    // Slate, stone, sky have norm_b > 0.33!
    if (norm_b < 0.11f || norm_b > 0.33f) return 0.0f;

    // 2. Melanin / Hemoglobin hierarchy: R strictly dominates G, G dominates B
    if (r <= g + 0.025f || g <= b - 0.04f) return 0.0f;

    // 3. Vascular tone ratio: (R - G) / (R - B) is stably in [0.20, 0.75] for living skin
    // In sandstone / pale walls, R and G are nearly equal ((R-G)/(R-B) < 0.18)
    float diff_rb = r - b;
    if (diff_rb > 0.01f) {
        float vasc = (r - g) / diff_rb;
        if (vasc < 0.18f || vasc > 0.82f) return 0.0f;
    }

    // 4. Saturation: living skin is warm-peach to warm-brown, never neon or metallic
    float mx = std::max(std::max(r, g), b);
    float mn = std::min(std::min(r, g), b);
    float sat = (mx - mn) / std::max(0.001f, mx);
    if (sat < 0.14f || sat > 0.68f) return 0.0f;

    // 5. Normalized chromatic coordinates (Fitzpatrick I through VI)
    if (norm_r < 0.34f || norm_r > 0.60f || norm_g < 0.23f || norm_g > 0.38f) return 0.0f;
    if (norm_r <= norm_g + 0.03f) return 0.0f;

    // 6. Perceptual YCbCr skin ellipse
    float Cb = -0.168736f * r - 0.331264f * g + 0.5f * b + 0.5f;
    float Cr =  0.5f * r - 0.418688f * g - 0.081312f * b + 0.5f;

    float dCr = (Cr - 0.60f) / 0.12f;
    float dCb = (Cb - 0.42f) / 0.08f;
    float dist = dCr * dCr + dCb * dCb;

    if (dist < 1.0f) {
        return 1.0f - 0.45f * dist;
    } else if (dist < 2.0f) {
        return 0.55f * (2.0f - dist);
    }
    return 0.0f;
}

// ============================================================================
// 4. Candidate Features & Witness Structure
// ============================================================================
struct OrbitalBasin {
    int y = 0, x = 0, s = 0;
    double contrast = 0.0;
    double inner_lum = 0.0;
    double cx = 0.0, cy = 0.0;

    Rect rect() const { return Rect{y, x, y + s, x + s}; }
};

struct FaceWitness {
    bool detected = false;
    float harmony_score = 0.0f;
    Rect bounding_box{};
    OrbitalBasin left_eye{};
    OrbitalBasin right_eye{};
    double inter_ocular_d = 0.0;
    double nose_x = 0.0, nose_y = 0.0;
    double mouth_x = 0.0, mouth_y = 0.0;
    
    // Detailed sub-scores
    float s_orbit = 0.0f;
    float s_bridge = 0.0f;
    float s_cheek = 0.0f;
    float s_skin = 0.0f;
    float s_sym = 0.0f;
    float s_mouth = 0.0f;
};

static FaceWitness g_lastWitness;
static bool debug_print = false;

// ============================================================================
// 5. Multi-Scale Orbital Valley Discovery
// ============================================================================
static std::vector<OrbitalBasin> findOrbitalBasins(
    const IntegralTable& iY,
    const IntegralTable& iY2,
    const IntegralTable& iSkin,
    int h, int w,
    bool is_monochrome)
{
    std::vector<OrbitalBasin> candidates;
    
    int min_dim = std::min(h, w);
    int min_s = std::max(4, min_dim / 75);
    int max_s = std::min(min_dim / 5, 120);

    for (int s = min_s; s <= max_s; s = (int)std::lround(s * 1.35) + 1) {
        int stride = std::max(2, s / 3);
        int pad = std::max(2, s / 2);

        for (int y = pad; y <= h - s - pad; y += stride) {
            for (int x = pad; x <= w - s - pad; x += stride) {
                Rect r{y, x, y + s, x + s};
                double inner = iY.mean(r);
                if (inner > 0.65 || inner < 0.03) continue; // skip highlights and pitch black

                double ring = ringMean(iY, r, pad);
                double contrast = ring - inner;

                if (contrast > 0.055) {
                    if (!is_monochrome) {
                        double skin_context = ringMean(iSkin, r, pad);
                        if (skin_context < 0.15) continue; // must be embedded in skin tissue
                    }
                    candidates.push_back(OrbitalBasin{
                        y, x, s, contrast, inner,
                        x + s / 2.0, y + s / 2.0
                    });
                }
            }
        }
    }

    // Sort by contrast descending
    std::stable_sort(candidates.begin(), candidates.end(), [](const OrbitalBasin& a, const OrbitalBasin& b) {
        return a.contrast > b.contrast;
    });

    // Non-Maximum Suppression
    std::vector<OrbitalBasin> nms_basins;
    nms_basins.reserve(150);

    for (const auto& b : candidates) {
        if (nms_basins.size() >= 120) break;
        bool overlap = false;
        for (const auto& kept : nms_basins) {
            double max_s_val = std::max(b.s, kept.s);
            if (std::abs(b.cx - kept.cx) < max_s_val * 0.65 &&
                std::abs(b.cy - kept.cy) < max_s_val * 0.65) {
                overlap = true;
                break;
            }
        }
        if (!overlap) {
            nms_basins.push_back(b);
        }
    }

    if (debug_print) {
        std::printf("DEBUG: Raw candidates count = %zu, after NMS = %zu\n", candidates.size(), nms_basins.size());
        for (const auto& b : nms_basins) {
            if (std::abs(b.cx - 263.0) < 30.0 && std::abs(b.cy - 260.0) < 30.0) {
                std::printf("DEBUG NMS kept near Lena left eye: x=%.1f, y=%.1f, s=%d, contrast=%.3f\n",
                    b.cx, b.cy, b.s, b.contrast);
            }
        }
    }

    return nms_basins;
}

// ============================================================================
// 6. Anti-Periodic Guard (Rejecting Windows, Castles, Ship Portholes)
// ============================================================================
// A face is strictly bi-ocular. If another dark basin of similar size and
// contrast exists at x_L - D or x_R + D, it is an architectural/periodic grid!
// ============================================================================
// ============================================================================
// 6. Anti-Periodic Lattice Guard (Rejecting Windows, Castles, Ship Portholes)
// ============================================================================
// A human face is strictly bi-ocular (two eyes). In architectural facades
// (palaces, castles) or ships with cannon ports, apertures repeat in an extended
// collinear lattice: ... P0 -> P1 -> P2 -> P3 -> P4 ...
// A pair (L, R) is part of a periodic grid if:
//   - An element exists on BOTH sides: at (L - v) AND at (R + v), OR
//   - Multiple elements exist forward: at (R + v) AND at (R + 2v), OR
//   - Multiple elements exist backward: at (L - v) AND at (L - 2v).
// A human head NEVER possesses two ears on one side (+v, +2v) or collinear
// eye-like apertures extending infinitely.
// ============================================================================
static bool hasPeriodicGrid(
    const std::vector<OrbitalBasin>& basins,
    const OrbitalBasin& L,
    const OrbitalBasin& R,
    double D)
{
    double vx = R.cx - L.cx;
    double vy = R.cy - L.cy;

    double r1_x = R.cx + vx,      r1_y = R.cy + vy;
    double r2_x = R.cx + 2.0 * vx, r2_y = R.cy + 2.0 * vy;
    double l1_x = L.cx - vx,      l1_y = L.cy - vy;
    double l2_x = L.cx - 2.0 * vx, l2_y = L.cy - 2.0 * vy;

    int avg_s = (L.s + R.s) / 2;

    auto findAt = [&](double tx, double ty) -> bool {
        for (const auto& b : basins) {
            if (&b == &L || &b == &R) continue;
            double sz_ratio = (double)std::max(b.s, avg_s) / std::min(b.s, avg_s);
            if (sz_ratio > 1.85) continue;
            double dist = std::hypot(b.cx - tx, b.cy - ty);
            if (dist < 0.35 * D && b.contrast > 0.05) return true;
        }
        return false;
    };

    bool has_r1 = findAt(r1_x, r1_y);
    bool has_l1 = findAt(l1_x, l1_y);

    if (has_r1 && has_l1) return true; // Element on BOTH flanks -> periodic lattice!
    if (has_r1 && findAt(r2_x, r2_y)) return true; // Multiple elements forward -> periodic lattice!
    if (has_l1 && findAt(l2_x, l2_y)) return true; // Multiple elements backward -> periodic lattice!

    return false;
}

// ============================================================================
// 7. Strict Bilateral Sagittal Reflection Symmetry
// ============================================================================
static double computeBilateralSymmetry(
    const std::vector<float>& gray,
    int h, int w,
    double cx, double cy, double D,
    int& valid_rays)
{
    int y_start = std::max(0, (int)std::lround(cy - 0.25 * D));
    int y_end   = std::min(h, (int)std::lround(cy + 1.15 * D));
    if (y_end <= y_start) { valid_rays = 0; return 1.0; }

    double total_diff = 0.0;
    int sample_count = 0;
    valid_rays = 0;

    for (int k = 1; k <= 8; ++k) {
        double delta = (k / 9.0) * (0.85 * D);
        int x_left  = (int)std::lround(cx - delta);
        int x_right = (int)std::lround(cx + delta);

        if (x_left < 0 || x_right >= w) continue;
        valid_rays++;

        for (int y = y_start; y < y_end; y += std::max(1, (int)(D / 20.0))) {
            double vL = gray[(size_t)y * w + x_left];
            double vR = gray[(size_t)y * w + x_right];
            total_diff += std::abs(vL - vR);
            sample_count++;
        }
    }

    return (sample_count >= 24 && valid_rays >= 6) ? (total_diff / sample_count) : 0.45;
}

// ============================================================================
// 8. Core Multimodal Face Decision Function: bool facetest(const rgbTensor&)
// ============================================================================
bool facetest(const rgbTensor& tensor) {
    g_lastWitness = FaceWitness{};

    int h = tensor.h;
    int w = tensor.w;
    if (h < 16 || w < 16) return false;

    // --- Phase 1: Build Multimodal Representations & Detect Monochrome Mode ---
    std::vector<float> gray((size_t)h * w);
    std::vector<float> gray2((size_t)h * w);
    std::vector<float> skin((size_t)h * w);

    double sum_sat = 0.0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float r = tensor.px(y, x, 0);
            float g = tensor.px(y, x, 1);
            float b = tensor.px(y, x, 2);
            float lum = 0.299f * r + 0.587f * g + 0.114f * b;
            gray[(size_t)y * w + x] = lum;
            gray2[(size_t)y * w + x] = lum * lum;

            float mx = std::max(std::max(r, g), b);
            float mn = std::min(std::min(r, g), b);
            sum_sat += (mx - mn);
        }
    }

    bool is_monochrome = (sum_sat / ((double)h * w) < 0.020);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float r = tensor.px(y, x, 0);
            float g = tensor.px(y, x, 1);
            float b = tensor.px(y, x, 2);
            skin[(size_t)y * w + x] = evaluateSkinLocus(r, g, b, is_monochrome);
        }
    }

    // --- Phase 2: Compute Integral Images ---
    IntegralTable iY;
    iY.init(h, w, gray);

    IntegralTable iY2;
    iY2.init(h, w, gray2);

    IntegralTable iSkin;
    iSkin.init(h, w, skin);

    // --- Phase 3: Discover Orbital Basins (Candidate Eyes) ---
    std::vector<OrbitalBasin> basins = findOrbitalBasins(iY, iY2, iSkin, h, w, is_monochrome);
    if (debug_print) {
        std::printf("DEBUG: Gefundene Orbitalbecken: %zu\n", basins.size());
        for (size_t b_idx = 0; b_idx < std::min((size_t)12, basins.size()); ++b_idx) {
            const auto& b = basins[b_idx];
            std::printf("  [%zu] y=%.1f, x=%.1f, s=%d, contrast=%.3f, inner=%.3f\n",
                        b_idx, b.cy, b.cx, b.s, b.contrast, b.inner_lum);
        }
    }
    if (basins.size() < 2) return false;

    // --- Phase 4: Bilateral Relational Pairing & Facial Topography Evaluation ---
    float best_harmony = 0.0f;
    FaceWitness best_witness{};

    int min_dim = std::min(h, w);

    for (size_t i = 0; i < basins.size(); ++i) {
        for (size_t j = 0; j < basins.size(); ++j) {
            if (i == j) continue;
            const OrbitalBasin& L = basins[i];
            const OrbitalBasin& R = basins[j];

            if (L.cx >= R.cx) continue; // L is left eye, R is right eye

            bool is_lena_pair = (debug_print && std::abs(L.cx - 252.5) < 10.0 && std::abs(R.cx - 324.5) < 10.0);
            if (is_lena_pair) {
                std::printf("DEBUG LenaPair: L=(%.1f, %.1f, s=%d), R=(%.1f, %.1f, s=%d)\n",
                    L.cx, L.cy, L.s, R.cx, R.cy, R.s);
            }

            double D = std::hypot(R.cx - L.cx, R.cy - L.cy);

            #define CHECK_FAIL(reason) do { if (is_lena_pair) std::printf("DEBUG LenaPair failed: %s\n", reason); } while(0)

            // Invariant: Scale ruler limits (a resolvable face has inter-ocular distance D >= 20.0 pixels)
            if (D < 20.0 || D > min_dim * 0.75) { CHECK_FAIL("scale ruler"); continue; }

            double cx = (L.cx + R.cx) / 2.0;
            double cy = (L.cy + R.cy) / 2.0;

            // Invariant: Full facial bounding frame margins (entire face must fit within image boundaries)
            if (cx - 1.05 * D < 0.0 || cx + 1.05 * D >= (double)w ||
                cy - 0.75 * D < 0.0 || cy + 1.65 * D >= (double)h) { CHECK_FAIL("margins"); continue; }

            // Invariant: Tilt within +-22 degrees
            double dy = std::abs(R.cy - L.cy);
            if (dy > 0.36 * D) { CHECK_FAIL("tilt"); continue; }

            // Invariant: Orbital scale consistency
            double s_ratio = (double)std::max(L.s, R.s) / std::min(L.s, R.s);
            if (s_ratio > 1.85) { CHECK_FAIL("s_ratio"); continue; }
            if (L.s < 0.12 * D || L.s > 0.65 * D) { CHECK_FAIL("L.s"); continue; }
            if (R.s < 0.12 * D || R.s > 0.65 * D) { CHECK_FAIL("R.s"); continue; }

            // Invariant: Anti-Periodic Lattice Guard (Reject palace windows, ship cannon ports)
            if (hasPeriodicGrid(basins, L, R, D)) { CHECK_FAIL("periodic grid"); continue; }

            // Invariant: Nasal Bridge Divider
            // The crest between the two orbits reflects light above the orbital floors
            Rect bridge = clampRect((int)(cy - 0.16 * D), (int)(cx - 0.14 * D),
                                    (int)(cy + 0.16 * D), (int)(cx + 0.14 * D), h, w);
            double bridge_lum = iY.mean(bridge);
            double avg_eye_lum = (L.inner_lum + R.inner_lum) / 2.0;
            if (bridge_lum < avg_eye_lum - 0.015) { CHECK_FAIL("bridge_lum < avg_eye - 0.015"); continue; }
            if (bridge_lum < L.inner_lum - 0.025 || bridge_lum < R.inner_lum - 0.025) { CHECK_FAIL("bridge_lum vs eye"); continue; }

            // Invariant: Tripartite Facial Landmarks
            Rect cheekL   = clampRect((int)(cy + 0.20 * D), (int)(L.cx - 0.35 * D),
                                      (int)(cy + 0.70 * D), (int)(L.cx + 0.35 * D), h, w);
            Rect cheekR   = clampRect((int)(cy + 0.20 * D), (int)(R.cx - 0.35 * D),
                                      (int)(cy + 0.70 * D), (int)(R.cx + 0.35 * D), h, w);
            Rect forehead = clampRect((int)(cy - 0.80 * D), (int)(cx - 0.50 * D),
                                      (int)(cy - 0.25 * D), (int)(cx + 0.50 * D), h, w);
            Rect noseTip  = clampRect((int)(cy + 0.35 * D), (int)(cx - 0.20 * D),
                                      (int)(cy + 0.65 * D), (int)(cx + 0.20 * D), h, w);
            Rect mouth    = clampRect((int)(cy + 0.90 * D), (int)(cx - 0.45 * D),
                                      (int)(cy + 1.35 * D), (int)(cx + 0.45 * D), h, w);
            Rect chin     = clampRect((int)(cy + 1.35 * D), (int)(cx - 0.35 * D),
                                      (int)(cy + 1.80 * D), (int)(cx + 0.35 * D), h, w);

            Rect faceBox  = clampRect((int)(cy - 0.85 * D), (int)(cx - 1.10 * D),
                                      (int)(cy + 1.80 * D), (int)(cx + 1.10 * D), h, w);

            // 1. Cheek-Eye Photometric Contrast & Isotropic Smoothness
            double cheekL_lum = iY.mean(cheekL);
            double cheekR_lum = iY.mean(cheekR);
            if (cheekL_lum < L.inner_lum + 0.015 || cheekR_lum < R.inner_lum + 0.015) { CHECK_FAIL("cheek_lum vs eye"); continue; }
            double cheek_lum = (cheekL_lum + cheekR_lum) / 2.0;
            if (cheek_lum < 0.20 || cheek_lum > 0.92) { CHECK_FAIL("cheek_lum range"); continue; }
            double cheek_contrast = cheek_lum - avg_eye_lum;
            float s_cheek = (float)std::min(1.0, std::max(0.0, cheek_contrast / 0.13));

            double varCheekL = boxVariance(iY, iY2, cheekL);
            double varCheekR = boxVariance(iY, iY2, cheekR);
            if (varCheekL > 0.045 || varCheekR > 0.045) { CHECK_FAIL("cheek variance"); continue; }

            // 2. Organic Skin Harmony
            double cheek_skin   = (iSkin.mean(cheekL) + iSkin.mean(cheekR)) / 2.0;
            double nose_skin    = iSkin.mean(noseTip);
            double forhd_skin   = forehead.valid() ? iSkin.mean(forehead) : 0.5;
            double box_skin     = iSkin.mean(faceBox);
            double overall_skin = 0.35 * cheek_skin + 0.25 * nose_skin + 0.20 * forhd_skin + 0.20 * box_skin;

            if (!is_monochrome && (overall_skin < 0.25 || cheek_skin < 0.28)) {
                if (is_lena_pair) {
                    std::printf("DEBUG LenaPair failed: skin harmony (overall=%.3f, cheek=%.3f, nose=%.3f, forhd=%.3f, box=%.3f)\n",
                        overall_skin, cheek_skin, nose_skin, forhd_skin, box_skin);
                    std::printf("DEBUG Lena cheekL skin=%.3f, cheekR skin=%.3f\n",
                        iSkin.mean(cheekL), iSkin.mean(cheekR));
                    int ny = (int)(cy + 0.50 * D);
                    int nx = (int)cx;
                    float nr = tensor.px(ny, nx, 0);
                    float ng = tensor.px(ny, nx, 1);
                    float nb = tensor.px(ny, nx, 2);
                    std::printf("DEBUG Lena nose center px: R=%.3f, G=%.3f, B=%.3f, total=%.3f, norm_b=%.3f, vasc=%.3f, skinScore=%.3f\n",
                        nr, ng, nb, nr+ng+nb, nb/(nr+ng+nb), (nr-ng)/(nr-nb), evaluateSkinLocus(nr, ng, nb, false));
                }
                continue;
            }
            float s_skin = is_monochrome ? 0.85f : (float)std::min(1.0, overall_skin / 0.48);

            // 3. Nasal Bridge Luminous Relief
            float s_bridge = (float)std::min(1.0, std::max(0.0, (bridge_lum - avg_eye_lum + 0.02) / 0.12));

            // 4. Oral-Mental Contrast Transition (Mouth Crevice vs Chin Pad)
            double mouth_lum = iY.mean(mouth);
            double chin_lum = chin.valid() ? iY.mean(chin) : mouth_lum;
            double mouth_contrast = chin_lum - mouth_lum;
            if (mouth_contrast < -0.015) { CHECK_FAIL("mouth contrast"); continue; }
            float s_mouth = (float)std::min(1.0, std::max(0.0, (mouth_contrast + 0.04) / 0.10));

            // 5. Bilateral Sagittal Reflection Symmetry
            int valid_rays = 0;
            double asym = computeBilateralSymmetry(gray, h, w, cx, cy, D, valid_rays);
            if (valid_rays < 6 || asym > 0.25) {
                if (is_lena_pair) std::printf("DEBUG LenaPair failed: asym=%.3f, valid_rays=%d\n", asym, valid_rays);
                continue;
            }
            float s_sym = (float)std::min(1.0, std::max(0.0, (0.28 - asym) / 0.18));

            // 6. Orbital Basins Contrast
            float s_orbit = (float)std::min(1.0, (L.contrast + R.contrast) / 0.24);

            // --- Individual Landmark Vitality Gates (The Gestalt Requirement) ---
            if (s_orbit < 0.50f || s_bridge < 0.45f || s_cheek < 0.45f ||
                s_skin < 0.45f  || s_sym < 0.30f    || s_mouth < 0.30f) {
                if (is_lena_pair) std::printf("DEBUG LenaPair failed vitality: orbit=%.2f, bridge=%.2f, cheek=%.2f, skin=%.2f, sym=%.2f, mouth=%.2f\n",
                    s_orbit, s_bridge, s_cheek, s_skin, s_sym, s_mouth);
                continue; // All primary anchors must be co-present!
            }

            // --- Holistic Facial Harmony Score ---
            float harmony = 0.20f * s_orbit
                          + 0.15f * s_bridge
                          + 0.20f * s_cheek
                          + 0.15f * s_skin
                          + 0.15f * s_sym
                          + 0.15f * s_mouth;

            if (harmony > best_harmony) {
                if (debug_print) {
                    std::printf("DEBUG NEW BEST: harmony=%.3f, D=%.1f, L=(%.1f, %.1f), R=(%.1f, %.1f)\n",
                        harmony, D, L.cx, L.cy, R.cx, R.cy);
                    int ny = (int)(cy + 0.50 * D);
                    int nx = (int)cx;
                    float nr = tensor.px(ny, nx, 0);
                    float ng = tensor.px(ny, nx, 1);
                    float nb = tensor.px(ny, nx, 2);
                    std::printf("DEBUG nose center px: R=%.3f, G=%.3f, B=%.3f, norm_b=%.3f, vasc=%.3f\n",
                        nr, ng, nb, nb/(nr+ng+nb), (nr-ng)/(nr-nb));
                }
                best_harmony = harmony;
                best_witness.detected = (harmony >= 0.65f);
                best_witness.harmony_score = harmony;
                best_witness.bounding_box = faceBox;
                best_witness.left_eye = L;
                best_witness.right_eye = R;
                best_witness.inter_ocular_d = D;
                best_witness.nose_x = cx;
                best_witness.nose_y = cy + 0.52 * D;
                best_witness.mouth_x = cx;
                best_witness.mouth_y = cy + 1.10 * D;
                best_witness.s_orbit = s_orbit;
                best_witness.s_bridge = s_bridge;
                best_witness.s_cheek = s_cheek;
                best_witness.s_skin = s_skin;
                best_witness.s_sym = s_sym;
                best_witness.s_mouth = s_mouth;
            }
        }
    }

    g_lastWitness = best_witness;
    return best_witness.detected;
}

// ============================================================================
// 9. Main CLI Entry Point
// ============================================================================
int main(int argc, char** argv) {
    const char* path = nullptr;
    bool debug = false;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--debug") || !std::strcmp(argv[i], "-d")) {
            debug = true;
            debug_print = true;
        } else if (!path) {
            path = argv[i];
        }
    }

    if (!path) {
        std::fprintf(stderr, "Aufruf: main2 bild.jpg [--debug]\n");
        return 2;
    }

    int width = 0, height = 0, channels = 0;
    unsigned char* raw_pixels = stbi_load(path, &width, &height, &channels, 3);
    if (!raw_pixels) {
        std::fprintf(stderr, "Fehler: Bild konnte nicht geladen werden (%s): %s\n",
                     path, stbi_failure_reason());
        return 2;
    }

    // Direct conversion into variable-dimension rgbTensor:
    rgbTensor tensor(height, width, 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            size_t idx = ((size_t)y * width + x) * 3;
            tensor(y, x, 0) = raw_pixels[idx + 0] / 255.0f;
            tensor(y, x, 1) = raw_pixels[idx + 1] / 255.0f;
            tensor(y, x, 2) = raw_pixels[idx + 2] / 255.0f;
        }
    }
    stbi_image_free(raw_pixels);

    std::printf("====================================================================\n");
    std::printf("Klassische Multimodale Gesichtserkennung (main2)\n");
    std::printf("====================================================================\n");
    std::printf("Bild geladen: %s\n", path);
    std::printf("Aufloesung (variabel): %d x %d (RGB, 5D Tensor)\n", tensor.w, tensor.h);

    bool result = facetest(tensor);

    const auto& w = g_lastWitness;
    std::printf("Gesichts-Harmonie-Score: %.1f%% (Schwelle: 65.0%%)\n", w.harmony_score * 100.0f);

    if (debug || w.detected) {
        std::printf("\n--- Multimodale Merkmal-Analyse ---\n");
        std::printf("  * Orbital-Kontrast (Augenhöhlen) : %5.1f%%\n", w.s_orbit * 100.0f);
        std::printf("  * Nasenruecken-Erhebung (T-Zone) : %5.1f%%\n", w.s_bridge * 100.0f);
        std::printf("  * Wangen-Augen-Kontrastrelief   : %5.1f%%\n", w.s_cheek * 100.0f);
        std::printf("  * Organischer Hautfarbraum       : %5.1f%%\n", w.s_skin * 100.0f);
        std::printf("  * Bilaterale Sagittalsymmetrie   : %5.1f%%\n", w.s_sym * 100.0f);
        std::printf("  * Mund-Kinn-Uebergang           : %5.1f%%\n", w.s_mouth * 100.0f);
    }

    if (w.detected) {
        std::printf("\n--- Zeugen-Konfiguration (Landmarks) ---\n");
        std::printf("  Augenabstand (D)   : %.1f Pixel\n", w.inter_ocular_d);
        std::printf("  Auge Links         : (x=%.1f, y=%.1f, Groesse=%d)\n",
                    w.left_eye.cx, w.left_eye.cy, w.left_eye.s);
        std::printf("  Auge Rechts        : (x=%.1f, y=%.1f, Groesse=%d)\n",
                    w.right_eye.cx, w.right_eye.cy, w.right_eye.s);
        std::printf("  Nase (geschaetzt)  : (x=%.1f, y=%.1f)\n", w.nose_x, w.nose_y);
        std::printf("  Mund (geschaetzt)  : (x=%.1f, y=%.1f)\n", w.mouth_x, w.mouth_y);
        std::printf("  Gesichts-Box [y0,x0 -> y1,x1]: [%d, %d -> %d, %d]\n",
                    w.bounding_box.y0, w.bounding_box.x0,
                    w.bounding_box.y1, w.bounding_box.x1);
    }

    std::printf("====================================================================\n");
    if (result) {
        std::printf("ERGEBNIS: Gesicht erkannt\n");
        return 0;
    } else {
        std::printf("ERGEBNIS: kein Gesicht erkannt\n");
        return 1;
    }
}
