// Media timing and native-coordinate drawing are independent of review labels.
export class ClipPlayer {
  constructor(video,changed,error) {
    this.video=video;this.changed=changed;this.error=error;this.info=null;this.generation=0;this.playEpoch=0;this.loading=false;
    video.addEventListener('timeupdate',()=>this.tick());
    video.addEventListener('play',()=>{const epoch=++this.playEpoch;this.changed();this.frameTick(epoch);});
    video.addEventListener('pause',()=>{this.playEpoch++;this.changed();});
    video.addEventListener('seeked',()=>{this.loading=false;this.changed();});
    video.addEventListener('loadeddata',()=>{if(!video.seeking){this.loading=false;this.changed();}});
    video.addEventListener('error',()=>{this.loading=false;this.changed();this.error('This browser could not play the original video. You can still open the exact frame to review.');});
  }
  async load(taskId) {
    const generation=++this.generation;this.pause();this.video.removeAttribute('src');this.info=null;this.loading=true;this.changed();
    const response=await fetch(`/api/clip/${taskId}`);if(!response.ok)throw Error('The source clip is unavailable.');
    const info=await response.json();if(generation!==this.generation)return;
    this.info=info;this.video.onloadedmetadata=()=>{if(generation===this.generation){this.video.currentTime=info.start_seconds;this.changed();}};
    this.video.src=info.url;this.video.load();this.changed();
  }
  pause(){this.video.pause();}
  async play(){if(!this.info)return;if(this.video.currentTime<this.info.start_seconds||this.video.currentTime>=this.info.end_seconds-.03)this.video.currentTime=this.info.start_seconds;await this.video.play();}
  seek(relative){if(!this.info)return;this.pause();this.loading=true;this.video.currentTime=Math.min(this.info.end_seconds-.025,Math.max(this.info.start_seconds,this.info.start_seconds+relative));this.changed();}
  tick(){if(this.info&&!this.video.paused&&this.video.currentTime>=this.info.end_seconds-.025)this.pause();this.changed();}
  frameTick(epoch){if(this.video.paused||epoch!==this.playEpoch)return;this.tick();if(this.video.requestVideoFrameCallback)this.video.requestVideoFrameCallback(()=>this.frameTick(epoch));}
  get relative(){return this.info?Math.max(0,this.video.currentTime-this.info.start_seconds):0;}
  get duration(){return this.info?this.info.end_seconds-this.info.start_seconds:6;}
}

export class StillView {
  constructor(canvas,options) {
    this.canvas=canvas;this.ctx=canvas.getContext('2d');this.options=options;this.view={x:0,y:0,w:1920,h:1080};this.cache=new Map();this.sequence=0;this.ready=false;this.pan=false;this.stroke=null;
    canvas.addEventListener('pointerdown',event=>this.down(event));canvas.addEventListener('pointermove',event=>this.move(event));
    canvas.addEventListener('pointerup',event=>this.up(event));canvas.addEventListener('pointercancel',()=>this.cancel());canvas.addEventListener('contextmenu',event=>event.preventDefault());
    canvas.addEventListener('wheel',event=>{if(event.ctrlKey||event.metaKey){event.preventDefault();this.zoom(event.deltaY<0?1.15:1/1.15,this.point(event));}},{passive:false});
  }
  async load(taskId,index) {
    const ticket=++this.sequence,key=`${taskId}:${index}`;this.ready=false;this.cancel();this.options.loading(true);this.abort?.abort();this.abort=new AbortController();
    try {
      let image=this.cache.get(key);
      if(!image){const response=await fetch(`/api/frame/${taskId}/${index}`,{signal:this.abort.signal});if(!response.ok){const value=await response.json();throw Error(value.error||'Exact frame unavailable');}image=await createImageBitmap(await response.blob());}
      if(ticket!==this.sequence){if(!this.cache.has(key))image.close();return false;}
      this.cache.delete(key);this.cache.set(key,image);this.bitmap=image;this.taskId=taskId;this.index=index;this.ready=true;
      while(this.cache.size>3){const oldest=this.cache.keys().next().value;this.cache.get(oldest).close();this.cache.delete(oldest);}
      this.options.loading(false);this.draw();return true;
    }catch(error){if(ticket===this.sequence){this.options.loading(false);if(error.name!=='AbortError')this.options.error(error.message);}return false;}
  }
  reset(){this.sequence++;this.abort?.abort();this.ready=false;this.cancel();this.bitmap=null;for(const image of this.cache.values())image.close();this.cache.clear();this.whole();this.ctx.clearRect(0,0,1920,1080);}
  whole(){this.view={x:0,y:0,w:1920,h:1080};this.draw();}
  focus(point,factor=2){if(!point)return this.whole();const w=1920/factor,h=1080/factor;this.view={x:Math.max(0,Math.min(1920-w,point[0]-w/2)),y:Math.max(0,Math.min(1080-h,point[1]-h/2)),w,h};this.draw();}
  zoom(factor,point){const v=this.view,p=point||[v.x+v.w/2,v.y+v.h/2],w=Math.max(240,Math.min(1920,v.w/factor)),h=w*1080/1920;this.view={x:Math.max(0,Math.min(1920-w,p[0]-(p[0]-v.x)*w/v.w)),y:Math.max(0,Math.min(1080-h,p[1]-(p[1]-v.y)*h/v.h)),w,h};this.draw();}
  point(event){const box=this.canvas.getBoundingClientRect();return [this.view.x+(event.clientX-box.left)/box.width*this.view.w,this.view.y+(event.clientY-box.top)/box.height*this.view.h];}
  inside(event){const b=this.canvas.getBoundingClientRect();return event.clientX>=b.left&&event.clientX<=b.right&&event.clientY>=b.top&&event.clientY<=b.bottom;}
  cancel(){this.stroke=null;}
  down(event) {
    if(!this.ready||!this.inside(event))return;this.canvas.focus();const point=this.point(event),mode=this.options.mode();
    if(this.pan||event.button===2){this.stroke={pan:true,client:[event.clientX,event.clientY],view:{...this.view}};}
    else if(mode==='focus'){this.options.focus(point);return;}
    else if(mode){this.stroke={key:mode,points:[point],before:structuredClone(this.options.marks()[mode]||[]),start:[event.clientX,event.clientY],dragged:false,taskId:this.taskId,index:this.index};}
    else return;
    event.preventDefault();this.canvas.setPointerCapture(event.pointerId);
  }
  move(event) {
    const stroke=this.stroke;if(!stroke)return;
    if(stroke.pan){const b=this.canvas.getBoundingClientRect(),v=stroke.view;this.view={...v,x:Math.max(0,Math.min(1920-v.w,v.x-(event.clientX-stroke.client[0])/b.width*v.w)),y:Math.max(0,Math.min(1080-v.h,v.y-(event.clientY-stroke.client[1])/b.height*v.h))};this.draw();return;}
    if(!this.inside(event))return;
    if(Math.hypot(event.clientX-stroke.start[0],event.clientY-stroke.start[1])>3)stroke.dragged=true;
    const point=this.point(event);if(Math.hypot(point[0]-stroke.points.at(-1)[0],point[1]-stroke.points.at(-1)[1])>=2)stroke.points.push(point);this.draw();
  }
  up(event) {
    const stroke=this.stroke;if(!stroke)return;this.stroke=null;
    if(!stroke.pan&&this.ready&&stroke.taskId===this.taskId&&stroke.index===this.index){let points=stroke.points;
      if(stroke.key==='landmark')points=[points.at(-1)];else if(!stroke.dragged)points=[...stroke.before,points[0]];
      this.options.commit(stroke.key,points,stroke.before);
    }
    if(this.canvas.hasPointerCapture(event.pointerId))this.canvas.releasePointerCapture(event.pointerId);this.draw();
  }
  draw() {
    if(!this.bitmap)return;const ctx=this.ctx,v=this.view,scale=1920/v.w;
    ctx.setTransform(1,0,0,1,0,0);ctx.drawImage(this.bitmap,v.x,v.y,v.w,v.h,0,0,1920,1080);ctx.setTransform(scale,0,0,scale,-v.x*scale,-v.y*scale);
    const marks=this.options.marks();for(const [key,color] of [['hand_boundary','#71d2ff'],['object_boundary','#ffd17a'],['landmark','#ff9eae']]){
      const points=this.stroke?.key===key?this.stroke.points:(marks[key]||[]);if(!points.length)continue;
      ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=4/scale;ctx.lineJoin='round';ctx.lineCap='round';ctx.beginPath();ctx.moveTo(...points[0]);for(const p of points.slice(1))ctx.lineTo(...p);if(this.options.closed()&&key!=='landmark'&&points.length>2)ctx.closePath();ctx.stroke();
      if(key==='landmark'||points.length<3)for(const p of points){ctx.beginPath();ctx.arc(...p,6/scale,0,Math.PI*2);ctx.fill();}
    }
    if(this.options.mode()==='focus'&&this.options.focusPoint()){
      const [x,y]=this.options.focusPoint();ctx.strokeStyle='#d5eee7';ctx.lineWidth=3/scale;ctx.setLineDash([10/scale,7/scale]);ctx.strokeRect(Math.max(0,Math.min(960,x-480)),Math.max(0,Math.min(540,y-270)),960,540);ctx.setLineDash([]);ctx.beginPath();ctx.arc(x,y,8/scale,0,Math.PI*2);ctx.stroke();
    }
    ctx.setTransform(1,0,0,1,0,0);
  }
}
