function pulse(options)
    arguments
        options.R double = 2;
        options.C double = 3;
        options.q double = 1;
        options.delta double = 1;
    end
    R = options.R;
    C = options.C;
    q = options.q;
    Tm = R * C;
    delta = options.delta * Tm;
    function du = du(t)
        du = zeros(size(t));
        du(t <= 0) = 0;
        idx = (t > 0) & (t <= delta);
        du(idx) = R * q / delta * (1 - exp(-t(idx) / Tm));
        idx = t > delta;
        du(idx) = R * q / delta * (exp(delta / Tm) - 1) .* exp(-t(idx) / Tm);
    end
    fplot(@(t) du(t), [0 50])
    grid on
end

