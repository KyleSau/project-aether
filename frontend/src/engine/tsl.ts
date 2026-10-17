// TSL is used untyped in shader-building modules (the strict generic typings of
// @types/three's TSL add no safety for procedural node graphs and slow development).
import * as tsl from 'three/tsl';
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const T: any = tsl;
export default T;
