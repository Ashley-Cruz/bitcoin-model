import React, { Fragment } from 'react';
import * as tf from '@tensorflow/tfjs';
import './Btc.css';
import figuras from './figuras.svg';
const Btc = () => {

    // Se llama la API con datos del Bitcoin
        
    function consultarAPI(){
        const url = `https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=98`;

        fetch(url)
            .then( respuesta => respuesta.json())
            .then( cotizacion => {
                generarHistorial(cotizacion.Data);
            });
    }

    consultarAPI();

    //Inicialización de variables

    let x = [];
    let xs = [];
    let y = [];
    let ys = [];
    let t = [];

    async function generarHistorial(data){
        for(let i of data["Data"]){
            await x.push(Math.log(i.close)*0.1);
            await y.push(Math.log(i.close)*0.1);
        }
        t = tf.tensor(x[98], [1,1]);
        x.pop();
        y.shift();
        
        //Entradas
        xs = tf.tensor(x, [98,1]);
        //Salidas
        ys = tf.tensor(y, [98,1]);

        generacionModelo(xs, ys, t)
            .then( () => {
                resultadoFinal(model.predict(t))
            });

    }

    function resultadoFinal(t){
        t.print();
    }

    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 1,
        inputShape: [1]
    }));

    //Se compila el modelo
    model.compile({
        loss: 'meanSquaredError',
        optimizer: 'sgd'
    });

    async function generacionModelo(xs, ys, t){
        for(let i=0; i<100; i++){
            const respuesta = await model.fit(xs, ys, {shuffle:true, epochs: 100});
            console.log(respuesta.history.loss[0]);
        }
    }

    return ( 
        <Fragment>
            <div className="contenedor-btc">
                <div className="contenedor-texto">
                    <h1 className="texto-principal">FORECAST BTC-USD</h1>
                    <p className="texto-secundario">To se the result, open the browser console</p>
                </div>
                <div className="contenedor-fig">
                    <img src={figuras} className="figuras" alt="figuras"></img>
                </div>
            </div>
        </Fragment>
     );
}
 
export default Btc;