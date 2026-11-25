function normalize(value) {
    return String(value || '').normalize('NFD').replace(/[\u0300-\u036f]/g, '')
        .toLowerCase().replace(/[^a-z0-9]/g, '');
}

function equipmentName(item) {
    return item?.name || (item?.signature != null ? String(item.signature) : 'unavailable');
}

export function createBikeResolver(model) {
    const frames = new Map();
    const wheels = new Map();
    const add = (index, key, item) => {
        key = normalize(key);
        if (!key) return;
        if (index.has(key) && index.get(key) !== item) index.set(key, null);
        else index.set(key, item);
    };
    for (const frame of model.frames) {
        add(frames, frame.name, frame);
        add(frames, frame.id, frame);
    }
    for (const [name, id] of [
        ['Specialized Tarmac SL8', 'SpecializedTarmacSL82024'],
        ['Specialized S-Works Shiv Disc', 'SpecializedShiv2019'],
        ['Brompton Brompton P Line', 'BromptonPLine2025'],
    ]) {
        const frame = model.frames.find(item => item.id === id);
        if (frame) add(frames, name, frame);
    }
    for (const wheel of model.wheels) {
        add(wheels, `${wheel.make} ${wheel.model}`, wheel);
        add(wheels, wheel.id, wheel);
        const brand = normalize(wheel.make);
        if (brand && wheel.id.startsWith(brand)) add(wheels, wheel.id.slice(brand.length), wheel);
    }
    const resolveWheel = item => {
        const asset = item?.imageName?.replace(/^Wheel_/i, '');
        return (asset && wheels.get(normalize(asset))) || wheels.get(normalize(item?.name));
    };
    return (equipment, level) => {
        const front = equipmentName(equipment?.wheelFront);
        const rear = equipmentName(equipment?.wheelRear);
        const wheels = equipment?.wheelFront?.name && equipment?.wheelRear?.name &&
            front === rear ? front : `front: ${front} / rear: ${rear}`;
        const label = `${equipmentName(equipment?.frame)} / ${wheels}; assumed L${level}`;
        if (!equipment?.frame || !equipment.wheelFront || !equipment.wheelRear) {
            return {label, error: 'Equipment unavailable (requires Sauce 2.4)'};
        }
        const frame = frames.get(normalize(equipment.frame.name));
        const wheelFront = resolveWheel(equipment.wheelFront);
        const wheelRear = resolveWheel(equipment.wheelRear);
        if (!frame || !wheelFront || !wheelRear) {
            return {label, error: `No physics stats for ${!frame ? 'frame' : 'wheels'}: ${label}`};
        }
        if (wheelFront !== wheelRear) {
            return {label, error: `Mixed wheelsets have no combined physics stats: ${label}`};
        }
        if (!Number.isInteger(level) || level < 0 || level > 5) {
            return {label, error: 'Upgrade assumption must be a level from 0 to 5'};
        }
        const wheel = wheelFront;
        return {
            frame, wheel, label,
            key: `${frame.id}/${wheel.id}/${level}`,
            frameBias: frame.cda[level],
            wheelBias: (frame.type || '').toUpperCase() === 'TT' ? wheel.cdaTt : wheel.cda,
            bikeWeightKg: (frame.wt[level] + wheel.wt) / 1000,
            bikeType: model.frameTypeToBikeType[frame.type] || 'road_bike',
            crrReduction: frame.crrReduction[level],
        };
    };
}
