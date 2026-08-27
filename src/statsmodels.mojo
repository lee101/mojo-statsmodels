from std.math import exp, log
from std.sys import simd_width_of

comptime W = simd_width_of[DType.float64]()
comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def ptr(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def dot(a: Ptr, b: Ptr, n: Int) -> Float64:
    var acc0 = SIMD[DType.float64, W](0.0)
    var acc1 = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + 2 * W <= n:
        acc0 += a.load[width=W](i) * b.load[width=W](i)
        acc1 += a.load[width=W](i + W) * b.load[width=W](i + W)
        i += 2 * W
    while i + W <= n:
        acc0 += a.load[width=W](i) * b.load[width=W](i)
        i += W
    var total = (acc0 + acc1).reduce_add()
    while i < n:
        total += a[i] * b[i]
        i += 1
    return total


def cholesky(a: Ptr, k: Int) -> Bool:
    for i in range(k):
        for j in range(i + 1):
            var value = a[i * k + j]
            for h in range(j):
                value -= a[i * k + h] * a[j * k + h]
            if i == j:
                if value <= 0.0:
                    return False
                a[i * k + i] = value ** 0.5
            else:
                a[i * k + j] = value / a[j * k + j]
    return True


def chol_solve(l: Ptr, b: Ptr, k: Int):
    for i in range(k):
        var value = b[i]
        for j in range(i):
            value -= l[i * k + j] * b[j]
        b[i] = value / l[i * k + i]
    for ri in range(k):
        var i = k - 1 - ri
        var value = b[i]
        for j in range(i + 1, k):
            value -= l[j * k + i] * b[j]
        b[i] = value / l[i * k + i]


def inverse_from_cholesky(l: Ptr, dst: Ptr, rhs: Ptr, k: Int):
    for col in range(k):
        for i in range(k):
            rhs[i] = 1.0 if i == col else 0.0
        chol_solve(l, rhs, k)
        for row in range(k):
            dst[row * k + col] = rhs[row]


def zero_system(gram: Ptr, rhs: Ptr, k: Int):
    for i in range(k * k):
        gram[i] = 0.0
    for i in range(k):
        rhs[i] = 0.0


def update_normal_equations(
    x: Ptr,
    y: Ptr,
    weights: Ptr,
    gram: Ptr,
    rhs: Ptr,
    start: Int,
    end: Int,
    k: Int,
):
    for row in range(start, end):
        var w = weights[row]
        var base = row * k
        var weighted_y = w * y[row]
        var i = 0
        while i + W <= k:
            rhs.store(
                i,
                rhs.load[width=W](i)
                + x.load[width=W](base + i) * weighted_y,
            )
            i += W
        while i < k:
            rhs[i] += x[base + i] * weighted_y
            i += 1
        for i in range(k):
            var wx = w * x[base + i]
            var gram_base = i * k
            var j = 0
            while j + W <= i + 1:
                gram.store(
                    gram_base + j,
                    gram.load[width=W](gram_base + j)
                    + x.load[width=W](base + j) * wx,
                )
                j += W
            while j <= i:
                gram[i * k + j] += wx * x[base + j]
                j += 1


def mirror_system(gram: Ptr, k: Int):
    for i in range(k):
        for j in range(i + 1, k):
            gram[i * k + j] = gram[j * k + i]


def normal_equations(
    x: Ptr, y: Ptr, weights: Ptr, gram: Ptr, rhs: Ptr, n: Int, k: Int
):
    zero_system(gram, rhs, k)
    update_normal_equations(x, y, weights, gram, rhs, 0, n, k)
    mirror_system(gram, k)


def solve_wls(
    covariance: Ptr, factor: Ptr, rhs: Ptr, beta: Ptr, k: Int
) -> Int:
    for i in range(k * k):
        factor[i] = covariance[i]
    if not cholesky(factor, k):
        return 0
    chol_solve(factor, rhs, k)
    for i in range(k):
        beta[i] = rhs[i]
    inverse_from_cholesky(factor, covariance, rhs, k)
    return 1


@export("mst_wls_fit")
def mst_wls_fit(
    x_addr: Int,
    y_addr: Int,
    weights_addr: Int,
    beta_addr: Int,
    covariance_addr: Int,
    factor_addr: Int,
    rhs_addr: Int,
    n: Int,
    k: Int,
) abi("C") -> Int:
    var x = ptr(x_addr)
    var y = ptr(y_addr)
    var weights = ptr(weights_addr)
    var beta = ptr(beta_addr)
    var covariance = ptr(covariance_addr)
    var factor = ptr(factor_addr)
    var rhs = ptr(rhs_addr)
    normal_equations(x, y, weights, covariance, rhs, n, k)
    return solve_wls(covariance, factor, rhs, beta, k)


@export("mst_linear_predict")
def mst_linear_predict(
    x_addr: Int, beta_addr: Int, dst_addr: Int, n: Int, k: Int
) abi("C"):
    var x = ptr(x_addr)
    var beta = ptr(beta_addr)
    var dst = ptr(dst_addr)
    for row in range(n):
        dst[row] = dot(x + row * k, beta, k)


def bounded_exp(value: Float64) -> Float64:
    if value < -30.0:
        return exp(-30.0)
    if value > 30.0:
        return exp(30.0)
    return exp(value)


def mean_and_weight(
    eta: Float64, family: Int
) -> Tuple[Float64, Float64, Float64]:
    if family == 0:
        return (eta, 1.0, 1.0)
    if family == 1:
        var mu: Float64
        if eta >= 0.0:
            var e = exp(-eta)
            mu = 1.0 / (1.0 + e)
        else:
            var e = exp(eta)
            mu = e / (1.0 + e)
        if mu < 1.0e-12:
            mu = 1.0e-12
        if mu > 1.0 - 1.0e-12:
            mu = 1.0 - 1.0e-12
        var derivative = mu * (1.0 - mu)
        return (mu, derivative, derivative)
    var mu = bounded_exp(eta)
    return (mu, mu, mu)


def glm_system(
    x: Ptr,
    y: Ptr,
    offset: Ptr,
    freq: Ptr,
    variance_weights: Ptr,
    beta: Ptr,
    gram: Ptr,
    rhs: Ptr,
    n: Int,
    k: Int,
    family: Int,
):
    for i in range(k * k):
        gram[i] = 0.0
    for i in range(k):
        rhs[i] = 0.0
    for row in range(n):
        var base = row * k
        var eta = offset[row] + dot(x + base, beta, k)
        var parts = mean_and_weight(eta, family)
        var mu = parts[0]
        var derivative = parts[1]
        var family_variance = parts[2]
        var w = freq[row] * variance_weights[row] * derivative * derivative / family_variance
        var z_minus_offset = eta + (y[row] - mu) / derivative - offset[row]
        var weighted_z = w * z_minus_offset
        var i = 0
        while i + W <= k:
            rhs.store(
                i,
                rhs.load[width=W](i)
                + x.load[width=W](base + i) * weighted_z,
            )
            i += W
        while i < k:
            rhs[i] += x[base + i] * weighted_z
            i += 1
        for i in range(k):
            var wx = w * x[base + i]
            var gram_base = i * k
            var j = 0
            while j + W <= i + 1:
                gram.store(
                    gram_base + j,
                    gram.load[width=W](gram_base + j)
                    + x.load[width=W](base + j) * wx,
                )
                j += W
            while j <= i:
                gram[gram_base + j] += wx * x[base + j]
                j += 1
    for i in range(k):
        for j in range(i + 1, k):
            gram[i * k + j] = gram[j * k + i]


@export("mst_glm_irls")
def mst_glm_irls(
    x_addr: Int,
    y_addr: Int,
    offset_addr: Int,
    freq_addr: Int,
    variance_weights_addr: Int,
    beta_addr: Int,
    covariance_addr: Int,
    factor_addr: Int,
    rhs_addr: Int,
    next_addr: Int,
    n: Int,
    k: Int,
    family: Int,
    max_iter: Int,
    tolerance: Float64,
) abi("C") -> Int:
    var x = ptr(x_addr)
    var y = ptr(y_addr)
    var offset = ptr(offset_addr)
    var freq = ptr(freq_addr)
    var variance_weights = ptr(variance_weights_addr)
    var beta = ptr(beta_addr)
    var covariance = ptr(covariance_addr)
    var factor = ptr(factor_addr)
    var rhs = ptr(rhs_addr)
    var next_beta = ptr(next_addr)
    var used_iterations = max_iter + 1
    for iteration in range(max_iter):
        glm_system(
            x, y, offset, freq, variance_weights, beta, covariance, rhs,
            n, k, family,
        )
        for i in range(k * k):
            factor[i] = covariance[i]
        if not cholesky(factor, k):
            return -1
        chol_solve(factor, rhs, k)
        var largest_change = 0.0
        var largest_value = 1.0
        for i in range(k):
            next_beta[i] = rhs[i]
            var change = abs(next_beta[i] - beta[i])
            if change > largest_change:
                largest_change = change
            var magnitude = abs(next_beta[i])
            if magnitude > largest_value:
                largest_value = magnitude
        for i in range(k):
            beta[i] = next_beta[i]
        if largest_change <= tolerance * largest_value:
            used_iterations = iteration + 1
            break
    glm_system(
        x, y, offset, freq, variance_weights, beta, covariance, rhs,
        n, k, family,
    )
    for i in range(k * k):
        factor[i] = covariance[i]
    if not cholesky(factor, k):
        return -1
    inverse_from_cholesky(factor, covariance, rhs, k)
    return used_iterations


@export("mst_arma_objective")
def mst_arma_objective(
    y_addr: Int,
    parameters_addr: Int,
    residual_addr: Int,
    derivative_addr: Int,
    gradient_addr: Int,
    n: Int,
    p: Int,
    q: Int,
    has_constant: Int,
) abi("C") -> Float64:
    var y = ptr(y_addr)
    var parameters = ptr(parameters_addr)
    var residual = ptr(residual_addr)
    var derivative = ptr(derivative_addr)
    var gradient = ptr(gradient_addr)
    var count = p + q + has_constant
    var burn = p if p > q else q
    for i in range(n):
        residual[i] = 0.0
    for i in range(n * count):
        derivative[i] = 0.0
    for h in range(count):
        gradient[h] = 0.0
    var squared_error = 0.0
    for t in range(burn, n):
        var parameter_offset = 0
        var prediction = 0.0
        if has_constant == 1:
            prediction = parameters[0]
            parameter_offset = 1
        for j in range(p):
            prediction += parameters[parameter_offset + j] * y[t - j - 1]
        for j in range(q):
            prediction += parameters[parameter_offset + p + j] * residual[t - j - 1]
        var error = y[t] - prediction
        residual[t] = error
        squared_error += error * error
        for h in range(count):
            var value = 0.0
            if has_constant == 1 and h == 0:
                value = -1.0
            elif h >= parameter_offset and h < parameter_offset + p:
                value = -y[t - (h - parameter_offset) - 1]
            elif h >= parameter_offset + p:
                value = -residual[t - (h - parameter_offset - p) - 1]
            for j in range(q):
                value -= parameters[parameter_offset + p + j] * derivative[
                    (t - j - 1) * count + h
                ]
            derivative[t * count + h] = value
            gradient[h] += error * value
    var effective_n = Float64(n - burn)
    if squared_error <= 1.0e-300:
        squared_error = 1.0e-300
    for h in range(count):
        gradient[h] *= effective_n / squared_error
    return 0.5 * effective_n * log(squared_error / effective_n)
