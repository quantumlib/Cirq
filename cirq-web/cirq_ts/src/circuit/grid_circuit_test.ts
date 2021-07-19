// Copyright 2021 The Cirq Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import {expect} from 'chai';
import {GridCircuit} from './grid_circuit';
import {GridCoord} from './components/types';

describe('GridCircuit', () => {
    // Test that we get three.js objects out of this. We don't
    // have to expose the ugly implementation details, we get a circuit
    // and it gives the three.js objects. Validation logic/what are the objects
    // Circuit ASCII diagram drawer in Cirq is good to look at. 
    
    describe('with an empty input and no moments', () => {
        // Fill in when building up testing
    });
    
    describe('with a 2x2 grid and 5 moments as input', () => {
        const twoByTwoGrid : GridCoord[] = [
            {'row': 0, 'col': 0},
            {'row': 0, 'col': 1},
            {'row': 1, 'col': 0},
            {'row': 1, 'col': 1},
        ]
        const circuit = new GridCircuit(5, twoByTwoGrid);
    });
});