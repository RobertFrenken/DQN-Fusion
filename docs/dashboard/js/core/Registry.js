/* Chart type registry */

const _registry = new Map();

export function register(name, ChartClass) {
    _registry.set(name, ChartClass);
}

export function get(name) {
    if (!_registry.has(name)) {
        throw new Error(`Chart type '${name}' not registered. Available: ${[..._registry.keys()]}`);
    }
    return _registry.get(name);
}

export function list() {
    return [..._registry.keys()];
}
