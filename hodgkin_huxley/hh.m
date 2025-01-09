function hh(options)
    arguments
        options.I_ext function_handle = @(t) 10 * (t > 5 & t < 30) + 2 * (t > 40 & t < 45) + 5 * (t > 60) % External current (uA/cm^2) as a step pulse
    end
    % Parameters
    C_m = 1;        % Membrane capacitance (uF/cm^2)
    g_Na = 120;     % Sodium conductance (mS/cm^2)
    g_K = 36;       % Potassium conductance (mS/cm^2)
    g_L = 0.3;      % Leak conductance (mS/cm^2)
    E_Na = 50;      % Sodium reversal potential (mV)
    E_K = -77;      % Potassium reversal potential (mV)
    E_L = -54.4;    % Leak reversal potential (mV)

    % Initial conditions
    V0 = -65; % Resting membrane potential (mV)
    m0 = alpha_m(V0) / (alpha_m(V0) + beta_m(V0));
    n0 = alpha_n(V0) / (alpha_n(V0) + beta_n(V0));
    h0 = alpha_h(V0) / (alpha_h(V0) + beta_h(V0));
    y0 = [V0, m0, n0, h0]; % Initial state vector
    I_ext = options.I_ext;

    % Time span
    tspan = [0 100]; % Time in ms

    % Solve the system of ODEs
    [t, y] = ode45(@(t, y) hh_equations(t, y, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_ext), tspan, y0);

    % Plot the membrane potential
    figure;
    tiledlayout(2,1)
    nexttile
    plot(t, y(:, 1), 'b', 'LineWidth', 1.5);
    ylabel('Membrane Potential (mV)');
    title('Hodgkin-Huxley Model');
    grid on;
    nexttile
    plot(t, arrayfun(I_ext, t), 'r', 'LineWidth',1.5);
    xlabel('Time (ms)');
    ylabel('Current (uA/cm^2)');
    title('External current');
    grid on;
end

% Hodgkin-Huxley equations
function dydt = hh_equations(t, y, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_ext)
    V = y(1); m = y(2); n = y(3); h = y(4);

    % Currents
    I_Na = g_Na * m^3 * h * (V - E_Na);
    I_K = g_K * n^4 * (V - E_K);
    I_L = g_L * (V - E_L);

    % Differential equations
    dVdt = (I_ext(t) - I_Na - I_K - I_L) / C_m;
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m;
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n;
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h;

    dydt = [dVdt; dmdt; dndt; dhdt];
end

% Rate functions
function val = alpha_m(V), val = (2.5 - 0.1 * (V + 65)) / (exp(2.5 - 0.1 * (V + 65)) - 1); end
function val = beta_m(V), val = 4 * exp(-(V + 65) / 18); end
function val = alpha_n(V), val = (0.1 - 0.01 * (V + 65)) / (exp(1 - 0.1 * (V + 65)) - 1); end
function val = beta_n(V), val = 0.125 * exp(-(V + 65) / 80); end
function val = alpha_h(V), val = 0.07 * exp(-(V + 65) / 20); end
function val = beta_h(V), val = 1 / (exp(3 - 0.1 * (V + 65)) + 1); end