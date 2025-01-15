function instant()
    % Parameters
    R = 2;
    C = 1;
    u_spike = -63;
    u_rest = -65;
    u_r = u_rest;
    tau_m = R * C;
    u0 = u_rest; % Initial condition for u(t)
    I = @(t) (t < 10 | t > 30) * 1.5; % Input current
    tspan = [0 50]; % Time range for simulation

    opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-8, 'Events', @(~, u) spikeEvent(u));

    % Initialize results
    % t_out = zeros(tspan(2) - tspan(1) + 1,1);
    t_out = [];
    u_out = [];

    % Loop over integration to handle reset
    t_start = tspan(1);
    while t_start < tspan(2)
        [t, u, te, ~, ~] = ode45(@(t, u) dudt(t, u), [t_start tspan(2)], u0, opts);

        % Append results

        t_out = [t_out; t];
        u_out = [u_out; u];

        % Check if spike event occurred
        if isempty(te)
            break; % No spike detected, terminate loop
        end

        % Reset u to u_rest and continue integration
        t_start = te(end); % Restart from the event time
        u0 = u_r;       % Reset u to resting potential
    end

    % Plot results
    figure;
    subplot(2, 1, 1);
    dudv = (u_out - u_rest)/(u_spike - u_rest);
    plot(t_out, dudv, 'LineWidth', 1.5, 'Color','r');
    xlabel('Time (ms)');
    ylabel('Membrane Potential u(t) (mV)');
    title('Membrane Potential Over Time');
    grid on;

    subplot(2, 1, 2);
    plot(t_out, arrayfun(I, t_out), 'LineWidth', 1.5);
    xlabel('Time (ms)');
    ylabel('Input Current I(t)');
    title('Input Current Over Time');
    grid on;

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
