

# Simulate
def trajectory_simgd(vec_fn, theta0, psi0,
                     nsteps=50, hs_g=0.1, hs_d=0.1):
    theta, psi = vec_fn.postprocess(theta0, psi0)
    thetas, psis = [theta], [psi]

    if isinstance(hs_g, float):
        hs_g = [hs_g] * nsteps
    if isinstance(hs_d, float):
        hs_d = [hs_d] * nsteps
    assert(len(hs_g) == nsteps)
    assert(len(hs_d) == nsteps)

    for h_g, h_d in zip(hs_g, hs_d):
        v1, v2 = vec_fn(theta, psi)
        theta += h_g * v1
        psi += h_d * v2
        theta, psi = vec_fn.postprocess(theta, psi)
        thetas.append(theta)
        psis.append(psi)

    return thetas, psis


def trajectory_altgd(vec_fn, theta0, psi0,
                     nsteps=50, hs_g=0.1, hs_d=0.1, gsteps=1, dsteps=1):
    theta, psi = vec_fn.postprocess(theta0, psi0)
    thetas, psis = [theta], [psi]

    if isinstance(hs_g, float):
        hs_g = [hs_g] * nsteps
    if isinstance(hs_d, float):
        hs_d = [hs_d] * nsteps
    assert(len(hs_g) == nsteps)
    assert(len(hs_d) == nsteps)

    for h_g, h_d in zip(hs_g, hs_d):
        for it in range(gsteps):
            v1, v2 = vec_fn(theta, psi)
            theta += h_g * v1
            theta, psi = vec_fn.postprocess(theta, psi)

        for it in range(dsteps):
            v1, v2 = vec_fn(theta, psi)
            psi += h_d * v2
            theta, psi = vec_fn.postprocess(theta, psi)
        thetas.append(theta)
        psis.append(psi)

    return thetas, psis
