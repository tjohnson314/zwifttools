const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function mapContext(dpr = 1) {
    const elements = new Map();
    const callbacks = {};
    const shapes = [];
    const ctx = {
        fillRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {}, fill() {},
        arc(x, y, radius) { shapes.push({kind:'circle', x, y, radius}); },
        rect(x, y, width, height) { shapes.push({kind:'square', x, y, width, height}); },
    };
    const canvas = {
        width:800*dpr, height:500*dpr, style:{}, getContext:()=>ctx,
        getBoundingClientRect:()=>({left:0, top:0, width:800, height:500}),
        addEventListener() {},
    };
    elements.set('map-canvas', canvas);
    elements.set('map-container', {getBoundingClientRect:()=>({width:800,height:500})});
    for (const id of ['map-scale', 'map-scale-label', 'map-scale-bar', 'map-boundary-tooltip',
        'show-map-boundaries', 'map-zoom-in', 'map-zoom-out']) {
        elements.set(id, {
            style:{}, checked:true, hidden:true, textContent:'',
            setAttribute(name, value) {this[name]=value;},
            addEventListener() {},
        });
    }
    const context = vm.createContext({
        document:{getElementById:id=>elements.get(id)},
        window:{devicePixelRatio:dpr, addEventListener:(name, callback)=>{callbacks[name]=callback;}},
        Image:function(){},
        raceData:{route_segments:[],finish_line_km:10},
        showRouteSegments:false, checkedRanks:new Set(), selectedRank:null,
        SEGMENT_COLORS:{sprint:'#4dd0e1',kom:'#ffb74d',segment:'#aed581'},
    });
    vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/js/map_canvas.js'),'utf8'),context);
    const source=fs.readFileSync(path.join(__dirname,'../static/js/race_replay.js'),'utf8');
    vm.runInContext(source.slice(source.indexOf('let mapState = null;')),context);
    context.config={lat_min:0,lat_max:.01,lng_min:0,lng_max:.02};
    context.initMap(context.config,null);
    return {context,elements,canvas,callbacks,shapes};
}

function segment(overrides={}) {
    return {
        name:'Volcano KOM', pass:1, lap:1,
        start_distance_km:2,end_distance_km:6,
        latlng:[[.005,.009],[.005,.011]], ...overrides,
    };
}

test('replay scale tracks zoom and pixel density and is visible without rider positions',()=>{
    for (const dpr of [1,2]) {
        const {context,elements}=mapContext(dpr);
        context.drawMap([]);
        assert.equal(elements.get('map-scale').hidden,false);
        const width=parseFloat(elements.get('map-scale-bar').style.width);
        const label=elements.get('map-scale-label').textContent;
        assert.equal(label,'500 m');
        const latitude=context.canvasToGps(16*dpr,500*dpr-16*dpr).lat;
        const expectedMetersPerPixel=.02/500*111320*Math.cos(latitude*Math.PI/180);
        assert.ok(Math.abs(width*expectedMetersPerPixel-500)<1e-6);
        vm.runInContext('mapState.viewScale = 5;',context);
        context.drawMap([]);
        assert.equal(elements.get('map-scale-label').textContent,'100 m');
        assert.ok(Math.abs(parseFloat(elements.get('map-scale-bar').style.width)-width)<1e-5);
    }
});

test('boundaries retain exact GPS positions, deduplicate laps and omit points beyond the finish',()=>{
    const {context}=mapContext();
    const source=[segment(),segment({lap:2}),segment({name:'Partial',start_distance_km:8,end_distance_km:12,
        latlng:[[.006,.009],[.006,.011]]})];
    const before=JSON.stringify(source);
    const boundaries=context.replaySegmentBoundaries(source,10);
    assert.equal(boundaries.length,3);
    assert.equal(boundaries[0].lat,.005);
    assert.equal(boundaries[0].lng,.009);
    assert.equal(boundaries[1].kind,'Finish');
    assert.equal(boundaries[0].labels.length,2);
    assert.ok(boundaries[0].labels[1].includes('lap 2'));
    assert.equal(boundaries[2].kind,'Start');
    assert.equal(JSON.stringify(source),before);
    assert.equal(context.replaySegmentBoundaries([segment({latlng:null})],10).length,0);
});

test('boundary shapes render independently of route overlays and respond to the visibility toggle',()=>{
    const {context,elements,shapes}=mapContext(2);
    context.boundaries=context.replaySegmentBoundaries([segment()],10);
    vm.runInContext('mapState.segmentBoundaries = boundaries;',context);
    context.drawMap([]);
    assert.deepEqual(shapes.map(shape=>shape.kind),['circle','square']);
    assert.equal(shapes[0].radius,10);
    assert.equal(shapes[1].width,20);
    elements.get('show-map-boundaries').checked=false;
    shapes.length=0;
    context.drawMap([]);
    assert.equal(shapes.length,0);
});

test('boundary hover identifies names and endpoints while panning hides the tooltip',()=>{
    const {context,elements}=mapContext(2);
    context.boundaries=context.replaySegmentBoundaries([segment({name:'Sprint <test>'})],10);
    vm.runInContext('mapState.segmentBoundaries = boundaries;',context);
    const point=context.gpsToCanvas(.005,.009);
    context.onMapMouseMove({clientX:point.x/2,clientY:point.y/2});
    assert.equal(elements.get('map-boundary-tooltip').hidden,false);
    assert.equal(elements.get('map-boundary-tooltip').textContent,'Sprint <test> - Start');
    context.onMapMouseDown({clientX:point.x/2,clientY:point.y/2});
    assert.equal(elements.get('map-boundary-tooltip').hidden,true);
    context.onMapMouseUp();
    elements.get('show-map-boundaries').checked=false;
    context.onMapMouseMove({clientX:point.x/2,clientY:point.y/2});
    assert.equal(elements.get('map-boundary-tooltip').hidden,true);
});

test('resize redraws map scale even while replay is paused',()=>{
    const {context,canvas,callbacks,elements}=mapContext();
    canvas.width=400;
    callbacks.resize();
    assert.equal(canvas.width,800);
    assert.equal(elements.get('map-scale').hidden,false);
});

test('opposite directions retain distinct starts and finishes at a shared arch',()=>{
    const {context}=mapContext();
    const boundaries=context.replaySegmentBoundaries([
        segment(),
        segment({name:'Volcano KOM Reverse',latlng:[[.005,.011],[.005,.009]]}),
    ],10);
    assert.equal(boundaries.length,4);
    assert.equal(boundaries.filter(p=>p.lng===.009).length,2);
});

test('offscreen boundaries are not drawn and panning updates their positions and the scale',()=>{
    const {context,elements,shapes}=mapContext();
    context.boundaries=context.replaySegmentBoundaries([segment()],10);
    vm.runInContext('mapState.segmentBoundaries = boundaries;',context);
    context.drawMap([]);
    const width=parseFloat(elements.get('map-scale-bar').style.width);
    shapes.length=0;
    vm.runInContext('mapState.viewLng = 1; mapState.viewLat = 60;',context);
    context.drawMap([]);
    assert.equal(shapes.length,0);
    assert.ok(parseFloat(elements.get('map-scale-bar').style.width)<width);
});

test('both replay templates load the shared scale helper and boundary controls',()=>{
    for(const filename of ['race_replay.html','ride_compare.html']) {
        const html=fs.readFileSync(path.join(__dirname,'../templates',filename),'utf8');
        assert.ok(html.includes('id="map-scale"'));
        assert.ok(html.includes('id="show-map-boundaries"'));
        assert.ok(html.includes('id="map-boundary-tooltip"'));
        assert.ok(html.indexOf('/static/js/map_canvas.js')<html.indexOf('/static/js/race_replay.js'));
    }
});
