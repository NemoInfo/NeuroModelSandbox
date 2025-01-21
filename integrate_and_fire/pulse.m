function pulse(options)
    arguments
        options.R double = 2;
        options.C double = 3;
        options.q double = 5e2;
        options.delta double = 1;
    end
    R = options.R;
    C = options.C;
    q = options.q;
    Tm = R * C;
    u_rest = -55;
    delta = options.delta * Tm;
    function du = du(t)
        du = zeros(size(t));
        du(t <= 0) = 0;
        idx = (t > 0) & (t <= delta);
        du(idx) = R * q / delta * (1 - exp(-t(idx) / Tm));
        idx = t > delta;
        du(idx) = R * q / delta * (exp(delta / Tm) - 1) .* exp(-t(idx) / Tm);
    end
    fplot(@(t) du(t) + u_rest , [0 50], LineWidth=2)
    xlabel('t (ms)', FontSize=20);
    ylabel('u(t) (mV)', FontSize=20);
    title('Membrane Potential Over Time', FontSize=20);
    grid on
end

