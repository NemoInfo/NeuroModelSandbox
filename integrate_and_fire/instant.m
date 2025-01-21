function instant()
    % Parameters
    R = 1e6;                % 1 MOhm
    C = 1e-9;               % 1 nF
    u_spike = -55e-3;       % -55 mV
    u_rest = -65e-3;        % -65 mV
    u_r = -70e-3;           % -70 mV
    tau_m = R * C;
    u0 = u_rest;
    I = @(t) (t < 10e-3 | t > 30e-3) * ...
        10.5e-9;            % 10.5 nA
    tspan = [0 50] * 1e-3;  % 0-50 ms

    opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-8, 'Events', @(~, u) spikeEvent(u));

    % Initialize results
    % t_out = zeros(tspan(2) - tspan(1) + 1,1);
    t_out = [];
    ts_spike = [];
    u_out = [];

    % Loop over integration to handle reset
    t_start = tspan(1);
    while t_start < tspan(2)
        [t, u, te, ~, ~] = ode45(@(t, u) dudt(t, u), [t_start tspan(2)], u0, opts);

        t_out = [t_out; t];
        ts_spike = [ts_spike; te];
        u_out = [u_out; u];

        if isempty(te)
            break; % No spike detected, terminate loop
        end

        % Reset u to u_r and continue integration
        t_start = te(end);
        u0 = u_r;
    end
    
    tiledlayout(3, 1);
    
    ax = nexttile;
    plot(t_out * 1e3, u_out * 1e3, 'LineWidth', 1.5, 'Color','r');
    ylabel('u(t) (mV)', FontSize=20);
    t = title('a)', FontSize=20, HorizontalAlignment='left');
    t.Position(1) = ax.XLim(1) - 5;
    grid on;


    tau_s = 0.002;
    tau_f = 0.0004;
    w = 1.5;
    ax = nexttile;
    plot(t_out * 1e3, arrayfun(@(t) psp(t), t_out), 'LineWidth', 1.5, 'Color','r');
    ylabel('PSP(t) (mV)', FontSize=20);
    t = title('b)', FontSize=20, HorizontalAlignment='left');
    t.Position(1) = ax.XLim(1) - 5;
    grid on;

    ax = nexttile;
    plot(t_out * 1e3, arrayfun(I, t_out) * 1e9, 'LineWidth', 1.5);
    ylabel('I(t) (nA)', FontSize=20);
    t = title('c)', FontSize=20, HorizontalAlignment='left');
    t.Position(1) = ax.XLim(1) - 5;
    grid on;

    xlabel('t (ms)', FontSize=20);
    
    function psp = psp(t)
        if any(ts_spike < t)
            t_spike = ts_spike(ts_spike < t);
            psp = sum(w*(exp((t_spike-t)/tau_s)-exp((t_spike-t)/tau_f)));
        else
            psp = 0;
        end
    end

    % Define the differential equation
    function dudt = dudt(t, u)
        dudt = (-(u - u_rest) + R * I(t)) / tau_m;
    end

    function [value, isterminal, direction] = spikeEvent(u)
        value = u - u_spike; % Detect when u exceeds u_spike
        isterminal = 1;      % Stop integration when event occurs
        direction = 1;       % Detect only upward crossings
    end
end
