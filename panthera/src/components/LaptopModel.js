import { useRef, Suspense } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { useGLTF, OrbitControls, Environment, ContactShadows } from "@react-three/drei";

function Laptop(props) {
  const { scene } = useGLTF("/model/laptop.glb");
  const ref = useRef();

  useFrame((state) => {
    const t = state.clock.getElapsedTime();
    ref.current.rotation.y = Math.sin(t * 0.3) * 0.15 + 0.3;
    ref.current.position.y = Math.sin(t * 0.5) * 0.05;
  });

  return <primitive ref={ref} object={scene} {...props} />;
}

const LaptopModel = () => {
  return (
    <div className="h-[500px] w-full lg:h-[400px] md:h-[350px] sm:h-[300px]">
      <Canvas
        camera={{ position: [0, 1.5, 4], fov: 45 }}
        gl={{ antialias: true, alpha: true }}
        style={{ background: "transparent" }}
      >
        <ambientLight intensity={0.5} />
        <directionalLight position={[5, 5, 5]} intensity={1} castShadow />
        <pointLight position={[-5, 3, -5]} intensity={0.4} color="#B63E96" />
        <pointLight position={[5, 3, -5]} intensity={0.3} color="#58E6D9" />

        <Suspense fallback={null}>
          <Laptop scale={1.2} position={[0, -0.5, 0]} />
          <ContactShadows
            position={[0, -1.2, 0]}
            opacity={0.4}
            scale={8}
            blur={2}
            far={3}
          />
          <Environment preset="city" />
        </Suspense>

        <OrbitControls
          enableZoom={false}
          enablePan={false}
          minPolarAngle={Math.PI / 3}
          maxPolarAngle={Math.PI / 2.2}
          autoRotate
          autoRotateSpeed={0.5}
        />
      </Canvas>
    </div>
  );
};

export default LaptopModel;
