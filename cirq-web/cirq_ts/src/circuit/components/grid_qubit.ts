import {BufferGeometry, Group, LineBasicMaterial, Vector3, Line,Texture, BoxGeometry, MeshBasicMaterial, Mesh, CylinderGeometry, SphereGeometry} from 'three';

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
    }


    private createLine(): Line {
        const material = new LineBasicMaterial({color: 'gray'});
        const points = [];
        points.push(new Vector3(this.x, 0, this.y));
        points.push(new Vector3(this.x, this.moments, this.y));

        const geometry = new BufferGeometry().setFromPoints(points);

        return new Line(geometry, material);
    }
    
    public addBox(moment: number) {
                
        var x = document.createElement("canvas")!;
        var xc = x.getContext("2d")!;
        x.width = x.height = 128;
        xc.fillStyle = "yellow";
        xc.fillRect(0, 0, x.width, x.height);
        xc.font = "50pt arial bold";
        xc.fillStyle = "black";
        xc.fillText('H', x.width/2 - 25, x.height/2 + 25);
        const map = new Texture(x);
        map.needsUpdate = true;

        const geometry = new BoxGeometry(0.5, 0.5, 0.5);
        const material = new MeshBasicMaterial( {map: map, color: 'yellow' } );
        const cube = new Mesh( geometry, material );
        cube.position.set(this.x, moment, this.y);
        this.add( cube );
    }

    public addCNOT(toX: number, toY: number, moment: number) {
        const geometry = new SphereGeometry(0.1, 32, 32);

        const material = new MeshBasicMaterial({color: 'black'});

        const sphere = new Mesh(geometry, material);
        sphere.position.set(this.x, moment, this.y);
        this.add(sphere);

        const line = new LineBasicMaterial({color: 'black'});
        const points = [new Vector3(this.x, moment, this.y), new Vector3(toX, moment, toY)];
        const geometry2 = new BufferGeometry().setFromPoints(points);
        this.add(new Line(geometry2, line));

        const geometry3 = new CylinderGeometry(0.3, 0.3, 0.1, 32);
        const material2 = new MeshBasicMaterial({color: 'black'});
        const cylinder = new Mesh(geometry3, material2);
        cylinder.position.set(toX, moment, toY);
        this.add(cylinder);
    }
}