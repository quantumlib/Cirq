import {BufferGeometry, Sprite, SpriteMaterial, Group, LineBasicMaterial, Vector3, Line,Texture, BoxGeometry, MeshBasicMaterial, Mesh, CylinderGeometry, SphereGeometry, DoubleSide, Sphere} from 'three';

export class GridQubit extends Group {
    public x: number;
    public y: number;
    public moments: number;

    constructor(x: number, y: number, moments: number) {
        super();

        this.x = x;
        this.y = y;
        this.moments = moments;
        this.add(this.createLine());
        this.add(this.createLocationLabel())
    }


    private createLine(): Line {
        const material = new LineBasicMaterial({color: 'gray'});
        const points = [];
        points.push(new Vector3(this.x, 0, this.y));
        points.push(new Vector3(this.x, this.moments, this.y));

        const geometry = new BufferGeometry().setFromPoints(points);

        return new Line(geometry, material);
    }
    
    private createLocationLabel(): Sprite {
        const CANVAS_SIZE = 128;
        const text = `(${this.x}, ${this.y})`;

        const canvas = document.createElement('canvas');
        canvas.width = CANVAS_SIZE;
        canvas.height = CANVAS_SIZE;
        // Allows us to keep track of what we're adding to the
        // canvas.
        canvas.textContent = text;
      
        const context = canvas.getContext('2d')!;
        context.fillStyle = '#000000';
        context.textAlign = 'center';
        context.font = '20px Arial';
        context.fillText(text, CANVAS_SIZE / 2, CANVAS_SIZE / 2);
      
        const map = new Texture(canvas);
        map.needsUpdate = true;
      
        const material =  new SpriteMaterial({
          map: map,
          transparent: true, // for a transparent canvas background
        });

        const sprite = new Sprite(material);
        sprite.position.copy(new Vector3(this.x, -0.6, this.y));
        return sprite;
    }

    public addSingleQubitGate(label: string, color: string, moment: number) {
        if (label === 'X') {
            const geometry = new CylinderGeometry(0.3, 0.3, 0.1, 32,1, true, 0, 2*Math.PI );
            const material = new MeshBasicMaterial({color: 'black', side: DoubleSide});
            const cylinder = new Mesh(geometry, material);
            cylinder.position.set(this.x, moment, this.y);
            this.add(cylinder);
            return;
        }

        var x = document.createElement("canvas")!;
        var xc = x.getContext("2d")!;
        x.width = x.height = 128;
        xc.fillStyle = "yellow";
        xc.fillRect(0, 0, x.width, x.height);
        xc.font = "50pt arial bold";
        xc.fillStyle = "black";
        xc.fillText(label, x.width/2 - 25, x.height/2 + 25);
        const map = new Texture(x);
        map.needsUpdate = true;

        const geometry = new BoxGeometry(0.5, 0.5, 0.5);
        const material = new MeshBasicMaterial( {map: map, color: color } );
        const cube = new Mesh( geometry, material );
        cube.position.set(this.x, moment, this.y);
        this.add( cube );
    }

    public addControl(moment: number) {
        const geometry = new SphereGeometry(0.1, 32, 32);
        const material = new MeshBasicMaterial({color: 'black'});
        const sphere = new Mesh(geometry, material);
        sphere.position.set(this.x, moment, this.y);
        this.add(sphere);
    }

    public addLineToQubit(x: number, y: number, moment: number) {
        const line = new LineBasicMaterial({color: 'black'});
        const points = [new Vector3(this.x, moment, this.y), new Vector3(x, moment, y)];
        const geometry = new BufferGeometry().setFromPoints(points);
        this.add(new Line(geometry, line));
    }
}